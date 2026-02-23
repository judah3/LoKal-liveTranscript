import React, { useEffect, useMemo, useRef, useState } from "react";
import type { TranscriptLine } from "../types/contracts";

const MAX_RENDERED_LINES = 14;

interface PanelProps {
  title: string;
  lines: TranscriptLine[];
  subtitle?: string;
  statusPill?: string;
  animateNewCards?: boolean;
  partial?: string;
  partialIsQuestion?: boolean;
  loadingRows?: number[];
  loadingLabel?: string;
  partialLoadingLabel?: string;
  captionMode?: boolean;
  speechActive?: boolean;
  currentFinalText?: string;
  currentFinalIsQuestion?: boolean;
  retainedFinalText?: string;
  retainedFinalFading?: boolean;
}

export function LivePanel({
  title,
  lines,
  subtitle,
  statusPill,
  animateNewCards = false,
  partial,
  partialIsQuestion = false,
  loadingRows = [],
  loadingLabel = "Loading...",
  partialLoadingLabel = "Transcribing...",
  captionMode = false,
  speechActive = false,
  currentFinalText = "",
  currentFinalIsQuestion = false,
  retainedFinalText = "",
  retainedFinalFading = false
}: PanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const hiddenCount = Math.max(0, lines.length - MAX_RENDERED_LINES);
  const visibleLines = hiddenCount > 0 ? lines.slice(lines.length - MAX_RENDERED_LINES) : lines;
  const visibleOffset = hiddenCount;
  const loadingRowSet = useMemo(() => new Set(loadingRows), [loadingRows]);
  const [animatedRows, setAnimatedRows] = useState<Set<number>>(new Set());
  const lastLineCountRef = useRef(lines.length);
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

  useEffect(() => {
    if (!animateNewCards) {
      lastLineCountRef.current = lines.length;
      return;
    }
    const prev = lastLineCountRef.current;
    const next = lines.length;
    lastLineCountRef.current = next;
    if (next <= prev) {
      return;
    }
    const newRows = new Set<number>();
    for (let i = prev; i < next; i += 1) {
      newRows.add(i);
    }
    setAnimatedRows((current) => new Set([...current, ...newRows]));
    const timer = setTimeout(() => {
      setAnimatedRows((current) => {
        const updated = new Set(current);
        for (const idx of newRows) {
          updated.delete(idx);
        }
        return updated;
      });
    }, 700);
    return () => clearTimeout(timer);
  }, [animateNewCards, lines.length]);

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-header-main">
          <span>{title}</span>
          {statusPill ? <span className="panel-status-pill">{statusPill}</span> : null}
        </div>
        {subtitle ? <div className="panel-header-subtitle">{subtitle}</div> : null}
      </header>
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
              <p className={partialIsQuestion ? "panel-partial caption-current question" : "panel-partial caption-current"}>
                <span className="panel-inline-loading" />
                {partial}
                <span className="panel-partial-label">{partialLoadingLabel}</span>
              </p>
            ) : (
              <p className={currentFinalIsQuestion ? "caption-current-final question" : "caption-current-final"}>
                {currentFinalText || "Waiting for input..."}
              </p>
            )}
          </div>
        ) : (
          <>
        {visibleLines.length > 0 ? (
          <div className="panel-text">
            {visibleLines.map((line, idx) => {
              const rowIndex = visibleOffset + idx;
              const loading = loadingRowSet.has(rowIndex);
              const text = line.text || (loading ? loadingLabel : "");
              const qa = parseQA(text);
              const rowClass = loading
                ? "panel-line loading"
                : line.isQuestion
                  ? "panel-line question"
                  : qa
                    ? "panel-line qa-card"
                    : "panel-line";
              const animateClass = animatedRows.has(rowIndex) ? "qa-card-enter" : "";
              return (
                <p key={rowIndex} className={`${rowClass} ${animateClass}`.trim()}>
                  {text ? renderFormattedLine(text) : "No entries yet."}
                </p>
              );
            })}
          </div>
        ) : (
          <p className="panel-text panel-empty">Waiting for input...</p>
        )}
        {partial ? (
          <p className={partialIsQuestion ? "panel-partial question" : "panel-partial"}>
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

function renderFormattedLine(text: string): React.ReactNode {
  const qa = parseQA(text);
  if (!qa) {
    return text;
  }
  const q = qa.q;
  const a = qa.a;
  return (
    <>
      <span className="qa-question-label">Q:</span>{" "}
      <span className="qa-question-text">{q}</span>
      {"\n"}
      <span className="qa-answer-label">A:</span>{" "}
      <span className="qa-answer-text">{a}</span>
    </>
  );
}

function parseQA(text: string): { q: string; a: string } | null {
  const match = text.match(/^Q:\s*([\s\S]*?)\nA:\s*([\s\S]*)$/);
  if (!match) {
    return null;
  }
  return { q: match[1].trim(), a: match[2].trim() };
}
