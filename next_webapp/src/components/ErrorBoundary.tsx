"use client";

import React from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  /** Optional custom fallback. Defaults to a generic inline message. */
  fallback?: React.ReactNode;
  /** Optional label included in the console log, to identify which boundary tripped. */
  name?: string;
}

interface ErrorBoundaryState {
  hasError: boolean;
}

/**
 * Catches render-time errors thrown by its child subtree (e.g. a widget
 * fed malformed data) so that one crashing section doesn't take down the
 * rest of the page. This is a plain React error boundary and can only be
 * a class component - React has no hook equivalent.
 *
 * Not a replacement for Next.js's route-level `error.tsx` (e.g.
 * src/app/[locale]/error.tsx), which already handles a whole route
 * segment crashing. Use this to wrap a specific risky section - a chart,
 * a map embed, rendered third-party/user content - inside an otherwise
 * healthy page.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(`[ErrorBoundary${this.props.name ? `:${this.props.name}` : ""}]`, error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        this.props.fallback ?? (
          <div className="flex min-h-[120px] items-center justify-center rounded-2xl border border-dashed border-red-200 bg-red-50 px-6 py-8 text-center dark:border-red-900 dark:bg-red-950/20">
            <p className="text-sm font-medium text-red-500 dark:text-red-400">
              Something went wrong displaying this section.
            </p>
          </div>
        )
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
