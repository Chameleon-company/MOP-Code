"use client"
import useThemeStore from "@/zustand/store"
import { MdModeNight } from "react-icons/md";
import { IoMdSunny } from "react-icons/io";

export default function ModeHandler() {

    const toggleTheme = useThemeStore((state) => state.toggleTheme);
    const theme = useThemeStore((state) => state.theme);

    return (<button className="mx-4 text-2xl" onClick={toggleTheme}> {theme == "light" ? <MdModeNight /> : <IoMdSunny className="text-slate-200"/>}</button>)
}