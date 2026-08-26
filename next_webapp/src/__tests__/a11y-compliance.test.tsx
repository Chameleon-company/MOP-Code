/** @jest-environment jsdom */

import React from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Users } from "lucide-react";
import Footer from "../components/Footer";
import CityMetricCard from "../components/CityMetricCard";
import LanguageDropdown from "../components/LanguageDropdown";

const mockPush = jest.fn();
const mockRefresh = jest.fn();

jest.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useFormatter: () => ({
    number: (value: number) => String(value),
  }),
}));

jest.mock("@/i18n-navigation", () => ({
  Link: ({ children, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a {...props}>{children}</a>
  ),
  usePathname: () => "/",
  useRouter: () => ({ push: mockPush, refresh: mockRefresh }),
}));

jest.mock("next/image", () => ({
  __esModule: true,
  default: ({ alt, ...props }: React.ImgHTMLAttributes<HTMLImageElement>) => (
    <img alt={alt} {...props} />
  ),
}));

beforeEach(() => {
  mockPush.mockClear();
  mockRefresh.mockClear();
  window.requestAnimationFrame = jest.fn(() => 1);
  window.cancelAnimationFrame = jest.fn();
});

describe("accessibility compliance", () => {
  it("exposes footer headings and navigation landmarks", () => {
    render(<Footer />);

    expect(screen.getByRole("heading", { level: 2, name: "Quick Links" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Connect" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 3, name: "Follow us" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Newsletter" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Quick Links" })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Connect" })).toBeInTheDocument();
  });

  it("hides CityMetricCard's decorative icons from assistive technology", () => {
    const { container } = render(
      <CityMetricCard
        metric={{
          id: "population",
          title: "Population",
          value: "2.3M",
          change: 2.5,
          icon: <Users />,
          category: "population",
        }}
      />,
    );

    expect(container.querySelector('[aria-hidden="true"] svg')).toBeInTheDocument();
    expect(container.querySelectorAll('svg[aria-hidden="true"]')).toHaveLength(1);
  });

  it("uses buttons for language-changing actions", () => {
    const { container } = render(<LanguageDropdown />);

    fireEvent.click(screen.getByRole("button", { name: "Language" }));

    expect(container.querySelectorAll("a")).toHaveLength(0);
    expect(screen.getAllByRole("button")).toHaveLength(9);

    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(mockPush).toHaveBeenCalledWith("/", { locale: "en" });
    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  it("opens LanguageDropdown from the keyboard and closes when focus leaves", async () => {
    const user = userEvent.setup();

    render(
      <>
        <LanguageDropdown />
        <button type="button">Outside control</button>
      </>,
    );

    const trigger = screen.getByRole("button", { name: "Language" });
    await act(async () => {
      await user.tab();
    });
    expect(trigger).toHaveFocus();
    await act(async () => {
      await user.keyboard("{Enter}");
    });

    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("button", { name: "English" })).toBeInTheDocument();

    await act(async () => {
      await user.tab();
    });
    expect(screen.getByRole("button", { name: "English" })).toHaveFocus();

    await act(async () => {
      for (let index = 0; index < 8; index += 1) {
        await user.tab();
      }
    });

    expect(screen.getByRole("button", { name: "Outside control" })).toHaveFocus();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("button", { name: "English" })).not.toBeInTheDocument();
  });
});
