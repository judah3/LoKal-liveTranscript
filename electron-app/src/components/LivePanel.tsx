import React, { useEffect, useMemo, useRef } from "react";

const MAX_RENDERED_LINES = 14;

interface PanelProps {
  title: string;
  lines: string[];
  partial?: string;
  loadingRows?: number[];
  loadingLabel?: string;
  partialLoadingLabel?: string;
  captionMode?: boolean;
  speechActive?: boolean;
  currentFinalText?: string;
  retainedFinalText?: string;
  retainedFinalFading?: boolean;
}

export function LivePanel({
  title,
  lines,
  partial,
  loadingRows = [],
  loadingLabel = "Loading...",
  partialLoadingLabel = "Transcribing...",
  captionMode = false,
  speechActive = false,
  currentFinalText = "",
  retainedFinalText = "",
  retainedFinalFading = false
}: PanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const hiddenCount = Math.max(0, lines.length - MAX_RENDERED_LINES);
  const visibleLines = hiddenCount > 0 ? lines.slice(lines.length - MAX_RENDERED_LINES) : lines;
  const visibleOffset = hiddenCount;
  const loadingRowSet = useMemo(() => new Set(loadingRows), [loadingRows]);
  const loadingRowsKey = loadingRows.join(",");

  useEffect(() => {
    if (!bodyRef.current) {
      return;
    }
    const el = bodyRef.current;
    const rafId = requestAnimationFrame(() => {
      el.scrollTop = el.scrollHeight;
    });
    return () => cancelAnimationFrame(rafId);
  }, [lines, partial, loadingRowsKey, captionMode, currentFinalText, retainedFinalText, retainedFinalFading]);

  return (
    <section className="panel">
      <header className="panel-header">{title}</header>
      <div className="panel-body" ref={bodyRef}>
        {speechActive ? (
          <div className="speech-active-indicator">
            <span className="speech-active-dot" />
            Speech detected
          </div>
        ) : null}
        {captionMode ? (
          <div className="caption-live">
            {retainedFinalText ? (
              <p className={retainedFinalFading ? "caption-retained fading" : "caption-retained"}>{retainedFinalText}</p>
            ) : null}
            {partial ? (
              <p className="panel-partial caption-current">
                <span className="panel-inline-loading" />
                {partial}
                <span className="panel-partial-label">{partialLoadingLabel}</span>
              </p>
            ) : (
              <p className="caption-current-final">{currentFinalText || "Waiting for input..."}</p>
            )}
          </div>
        ) : (
          <>
        {visibleLines.length > 0 ? (
          <div className="panel-text">
            {visibleLines.map((line, idx) => {
              const rowIndex = visibleOffset + idx;
              const loading = loadingRowSet.has(rowIndex);
              const text = line || (loading ? loadingLabel : "");
              return (
                <p key={rowIndex} className={loading ? "panel-line loading" : "panel-line"}>
                  {text || "\u00A0"}
                </p>
              );
            })}
          </div>
        ) : (
          <p className="panel-text">Waiting for input...</p>
        )}
        {partial ? (
          <p className="panel-partial">
            <span className="panel-inline-loading" />
            {partial}
            <span className="panel-partial-label">{partialLoadingLabel}</span>
          </p>
        ) : null}
          </>
        )}
      </div>
    </section>
  );
}
