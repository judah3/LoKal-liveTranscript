using System.Text.Json;
using AudioCaptureService;

var jsonOptions = new JsonSerializerOptions
{
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
};

await using var streamer = new AudioLoopbackStreamer(message =>
{
    Console.Error.WriteLine($"[{DateTime.UtcNow:O}] {message}");
});

while (true)
{
    var line = await Console.In.ReadLineAsync();
    if (line is null)
    {
        break;
    }

    RequestEnvelope? request;
    try
    {
        request = JsonSerializer.Deserialize<RequestEnvelope>(line, jsonOptions);
    }
    catch (Exception ex)
    {
        await WriteAsync(new ResponseEnvelope
        {
            RequestId = "unknown",
            Type = "error",
            Error = $"Invalid JSON: {ex.Message}"
        });
        continue;
    }

    if (request is null)
    {
        continue;
    }

    try
    {
        switch (request.Type)
        {
            case "ping":
                await WriteAsync(new ResponseEnvelope
                {
                    RequestId = request.RequestId,
                    Type = request.Type,
                    Payload = new { ok = true }
                });
                break;

            case "list_devices":
                var devices = streamer.ListOutputDevices();
                await WriteAsync(new ResponseEnvelope
                {
                    RequestId = request.RequestId,
                    Type = request.Type,
                    Payload = devices
                });
                break;

            case "start":
                var deviceId = request.Payload.GetProperty("deviceId").GetString();
                var backendUrl = request.Payload.GetProperty("backendUrl").GetString();

                if (string.IsNullOrWhiteSpace(deviceId) || string.IsNullOrWhiteSpace(backendUrl))
                {
                    throw new InvalidOperationException("start requires deviceId and backendUrl");
                }

                await streamer.StartAsync(deviceId, new Uri(backendUrl), CancellationToken.None);
                await WriteAsync(new ResponseEnvelope
                {
                    RequestId = request.RequestId,
                    Type = request.Type,
                    Payload = new { started = true }
                });
                await EmitStatusAsync(true, null);
                break;

            case "stop":
                await streamer.StopAsync();
                await WriteAsync(new ResponseEnvelope
                {
                    RequestId = request.RequestId,
                    Type = request.Type,
                    Payload = new { stopped = true }
                });
                await EmitStatusAsync(false, null);
                break;

            case "shutdown":
                await streamer.StopAsync();
                await WriteAsync(new ResponseEnvelope
                {
                    RequestId = request.RequestId,
                    Type = request.Type,
                    Payload = new { shutdown = true }
                });
                return;

            default:
                throw new InvalidOperationException($"Unknown request type: {request.Type}");
        }
    }
    catch (Exception ex)
    {
        await WriteAsync(new ResponseEnvelope
        {
            RequestId = request.RequestId,
            Type = request.Type,
            Error = ex.Message
        });
        await EmitStatusAsync(false, ex.Message);
    }
}

async Task EmitStatusAsync(bool listening, string? error)
{
    await WriteAsync(new ResponseEnvelope
    {
        RequestId = string.Empty,
        Type = "status",
        Payload = new { listening, error }
    });
}

async Task WriteAsync(ResponseEnvelope response)
{
    var json = JsonSerializer.Serialize(response, jsonOptions);
    await Console.Out.WriteLineAsync(json);
    await Console.Out.FlushAsync();
}
