using System.Net.WebSockets;
using NAudio.CoreAudioApi;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;

namespace AudioCaptureService;

public sealed class AudioLoopbackStreamer : IAsyncDisposable
{
    private const int TargetSampleRate = 16000;
    private const int TargetChannels = 1;
    private const int NormalChunkMillis = 60;
    private const int FastChunkMillis = 25;
    private const int NormalSamplesPerChunk = TargetSampleRate * NormalChunkMillis / 1000;
    private const int FastSamplesPerChunk = TargetSampleRate * FastChunkMillis / 1000;
    private const float HighSpeechRmsThreshold = 0.0035f;
    private const int EnterFastModeStreak = 2;
    private const int ExitFastModeStreak = 8;
    private const int ReconnectDelayMillis = 400;
    private const int ConnectTimeoutMillis = 4000;

    private readonly Action<string> _log;
    private readonly SemaphoreSlim _gate = new(1, 1);

    private WasapiLoopbackCapture? _capture;
    private BufferedWaveProvider? _bufferedProvider;
    private ISampleProvider? _sampleProvider;
    private ClientWebSocket? _webSocket;
    private Uri? _backendUri;
    private CancellationTokenSource? _cts;
    private Task? _senderTask;
    private DateTime _lastDisconnectLogUtc = DateTime.MinValue;

    public bool IsRunning => _capture is not null && _webSocket is not null;

    public AudioLoopbackStreamer(Action<string> log)
    {
        _log = log;
    }

    public IReadOnlyList<DeviceInfo> ListOutputDevices()
    {
        using var enumerator = new MMDeviceEnumerator();
        var devices = enumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active);

        return devices
            .Select(d => new DeviceInfo { Id = d.ID, Name = d.FriendlyName })
            .ToArray();
    }

    public async Task StartAsync(string deviceId, Uri backendUri, CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken);
        try
        {
            await StopInternalAsync();

            using var enumerator = new MMDeviceEnumerator();
            using var device = enumerator.EnumerateAudioEndPoints(DataFlow.Render, DeviceState.Active)
                .FirstOrDefault(d => string.Equals(d.ID, deviceId, StringComparison.OrdinalIgnoreCase));

            if (device is null)
            {
                throw new InvalidOperationException($"Device not found: {deviceId}");
            }

            _capture = new WasapiLoopbackCapture(device);
            _bufferedProvider = new BufferedWaveProvider(_capture.WaveFormat)
            {
                DiscardOnBufferOverflow = true,
                BufferDuration = TimeSpan.FromSeconds(2)
            };

            var inputSampleProvider = _bufferedProvider.ToSampleProvider();
            var monoProvider = ConvertToMono(inputSampleProvider, _capture.WaveFormat.Channels);
            _sampleProvider = new WdlResamplingSampleProvider(monoProvider, TargetSampleRate);

            _capture.DataAvailable += OnDataAvailable;
            _capture.RecordingStopped += (_, e) =>
            {
                if (e.Exception is not null)
                {
                    _log($"Capture stopped with error: {e.Exception.Message}");
                }
            };

            _backendUri = backendUri;
            await EnsureConnectedAsync(cancellationToken);

            _cts = new CancellationTokenSource();
            _senderTask = Task.Run(() => SendLoopAsync(_cts.Token));

            _capture.StartRecording();
            _log("Capture started");
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task StopAsync()
    {
        await _gate.WaitAsync();
        try
        {
            await StopInternalAsync();
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task StopInternalAsync()
    {
        _cts?.Cancel();

        if (_capture is not null)
        {
            _capture.StopRecording();
            _capture.Dispose();
            _capture = null;
        }

        if (_senderTask is not null)
        {
            try
            {
                await _senderTask;
            }
            catch (OperationCanceledException)
            {
            }
            catch (WebSocketException ex)
            {
                _log($"Sender loop stopped due to websocket close: {ex.Message}");
            }
            catch (Exception ex)
            {
                _log($"Sender loop stopped with error: {ex.Message}");
            }
            _senderTask = null;
        }

        if (_webSocket is not null)
        {
            if (_webSocket.State is WebSocketState.Open or WebSocketState.CloseReceived)
            {
                try
                {
                    await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "stop", CancellationToken.None);
                }
                catch (WebSocketException ex)
                {
                    _log($"WebSocket close handshake skipped: {ex.Message}");
                }
            }
            _webSocket.Dispose();
            _webSocket = null;
        }

        _cts?.Dispose();
        _cts = null;
        _backendUri = null;
        _bufferedProvider = null;
        _sampleProvider = null;
    }

    private async Task SendLoopAsync(CancellationToken cancellationToken)
    {
        if (_sampleProvider is null || _webSocket is null)
        {
            return;
        }

        var floatChunk = new float[NormalSamplesPerChunk];
        var useFastChunking = false;
        var highSpeechStreak = 0;
        var lowSpeechStreak = 0;
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                if (_webSocket is null || _webSocket.State != WebSocketState.Open)
                {
                    await EnsureConnectedAsync(cancellationToken);
                    if (_webSocket is null || _webSocket.State != WebSocketState.Open)
                    {
                        await Task.Delay(ReconnectDelayMillis, cancellationToken);
                        continue;
                    }
                }

                var targetSamples = useFastChunking ? FastSamplesPerChunk : NormalSamplesPerChunk;
                var read = _sampleProvider.Read(floatChunk, 0, targetSamples);
                if (read < targetSamples)
                {
                    await Task.Delay(5, cancellationToken);
                    continue;
                }

                var rms = ComputeRms(floatChunk, read);
                if (rms >= HighSpeechRmsThreshold)
                {
                    highSpeechStreak++;
                    lowSpeechStreak = 0;
                }
                else
                {
                    lowSpeechStreak++;
                    highSpeechStreak = 0;
                }

                if (!useFastChunking && highSpeechStreak >= EnterFastModeStreak)
                {
                    useFastChunking = true;
                    highSpeechStreak = 0;
                }
                else if (useFastChunking && lowSpeechStreak >= ExitFastModeStreak)
                {
                    useFastChunking = false;
                    lowSpeechStreak = 0;
                }

                var bytes = new byte[targetSamples * sizeof(float)];
                Buffer.BlockCopy(floatChunk, 0, bytes, 0, bytes.Length);
                await _webSocket.SendAsync(bytes, WebSocketMessageType.Binary, true, cancellationToken);
                await Task.Delay(useFastChunking ? FastChunkMillis : NormalChunkMillis, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (WebSocketException ex)
            {
                // Remote close-handshake race is common during backend restart/teardown.
                // Keep the loop alive and reconnect without flooding logs.
                if (!IsExpectedClose(ex) && ShouldLogDisconnect())
                {
                    _log($"Audio websocket disconnected: {ex.Message}");
                }

                _webSocket?.Dispose();
                _webSocket = null;
                await Task.Delay(ReconnectDelayMillis, cancellationToken);
            }
        }
    }

    private async Task EnsureConnectedAsync(CancellationToken cancellationToken)
    {
        if (_backendUri is null)
        {
            return;
        }

        if (_webSocket is not null && _webSocket.State == WebSocketState.Open)
        {
            return;
        }

        _webSocket?.Dispose();
        _webSocket = new ClientWebSocket();
        _webSocket.Options.KeepAliveInterval = TimeSpan.FromSeconds(8);

        using var timeoutCts = new CancellationTokenSource(ConnectTimeoutMillis);
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);
        await _webSocket.ConnectAsync(_backendUri, linked.Token);
        _log("Audio websocket connected");
    }

    private static float ComputeRms(float[] samples, int count)
    {
        var energy = 0.0f;
        for (var i = 0; i < count; i++)
        {
            var s = samples[i];
            energy += s * s;
        }
        return MathF.Sqrt(energy / Math.Max(count, 1));
    }

    private bool ShouldLogDisconnect()
    {
        var now = DateTime.UtcNow;
        if ((now - _lastDisconnectLogUtc).TotalSeconds < 5)
        {
            return false;
        }
        _lastDisconnectLogUtc = now;
        return true;
    }

    private static bool IsExpectedClose(WebSocketException ex)
    {
        return ex.WebSocketErrorCode is WebSocketError.ConnectionClosedPrematurely
            or WebSocketError.InvalidState
            or WebSocketError.NotAWebSocket;
    }

    private void OnDataAvailable(object? sender, WaveInEventArgs e)
    {
        _bufferedProvider?.AddSamples(e.Buffer, 0, e.BytesRecorded);
    }

    private static ISampleProvider ConvertToMono(ISampleProvider provider, int channels)
    {
        if (channels == 1)
        {
            return provider;
        }

        if (channels == 2)
        {
            return new StereoToMonoSampleProvider(provider)
            {
                LeftVolume = 0.5f,
                RightVolume = 0.5f
            };
        }

        return new MultiChannelToMonoSampleProvider(provider, channels);
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync();
        _gate.Dispose();
    }

    private sealed class MultiChannelToMonoSampleProvider : ISampleProvider
    {
        private readonly ISampleProvider _source;
        private readonly float[] _buffer;
        private readonly int _channels;

        public MultiChannelToMonoSampleProvider(ISampleProvider source, int channels)
        {
            _source = source;
            _channels = channels;
            _buffer = new float[4096 * channels];
            WaveFormat = WaveFormat.CreateIeeeFloatWaveFormat(source.WaveFormat.SampleRate, 1);
        }

        public int Read(float[] buffer, int offset, int count)
        {
            var required = count * _channels;
            var read = _source.Read(_buffer, 0, required);
            var frames = read / _channels;
            for (var i = 0; i < frames; i++)
            {
                float sum = 0;
                for (var c = 0; c < _channels; c++)
                {
                    sum += _buffer[i * _channels + c];
                }
                buffer[offset + i] = sum / _channels;
            }

            return frames;
        }

        public WaveFormat WaveFormat { get; }
    }
}
