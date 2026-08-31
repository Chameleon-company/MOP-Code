/** @jest-environment jsdom */

import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import PrivacyPolicy from "../app/[locale]/privacypolicy/page";
import html2canvas from "html2canvas";

const addImage = jest.fn();
const addPage = jest.fn();
const save = jest.fn();
const mockCapturedText = jest.fn();

jest.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

jest.mock("../components/Header", () => () => <header>Header</header>);
jest.mock("../components/Footer", () => () => <footer>Footer</footer>);
jest.mock("html2canvas", () => jest.fn());
jest.mock("jspdf", () =>
  jest.fn().mockImplementation(() => ({
    internal: { pageSize: { getWidth: () => 210, getHeight: () => 297 } },
    addImage,
    addPage,
    save,
  }))
);

describe("Privacy Policy PDF export", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.requestAnimationFrame = (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    };
    Object.defineProperty(document, "fonts", {
      configurable: true,
      value: { ready: Promise.resolve() },
    });
    (html2canvas as jest.Mock).mockImplementation((element: HTMLElement) => {
      mockCapturedText(element.textContent);
      return Promise.resolve({
        width: 800,
        height: 2400,
        toDataURL: () => "data:image/png;base64,pdf",
      });
    });
  });

  it("expands every section and adds enough A4 pages for the complete canvas", async () => {
    render(<PrivacyPolicy />);

    fireEvent.click(screen.getByRole("button", { name: "Download PDF" }));

    await waitFor(() => expect(html2canvas).toHaveBeenCalledTimes(1));
    expect(mockCapturedText).toHaveBeenCalledWith(
      expect.stringMatching(/p1[\s\S]*p2[\s\S]*p3[\s\S]*p4[\s\S]*p5[\s\S]*p6/)
    );

    expect(addPage).toHaveBeenCalledTimes(2);
    expect(addImage).toHaveBeenCalledTimes(3);
    expect(save).toHaveBeenCalledWith("privacy-policy.pdf");
  });
});
