import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./globals.css";
import "./research-views.css";
import "./workbench.css";
import "./dream-theme.css";
import "./editorial.css";
import "./community.css";
import Home from "./page";

createRoot(document.getElementById("root")!).render(<StrictMode><Home /></StrictMode>);
