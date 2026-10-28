"""
Z.E.U.S. - Zone Evaluation and Utility System
Adani University | Faculty of Engineering Sciences and Technology (FEST)
Department of Computer Science and Engineering (AI-ML)
Course Code: ECSCT24301 - Foundation of AI
Team Members:
  1. Patel Meet (Enrollment No: 2410111108)
  2. Patel Sujal (Enrollment No: 2410111124)
  3. Somani Swayam (Enrollment No: 2410111163)

Core AI Architecture:
  Stage 1: Probabilistic Inference (Unit 3) - Bayesian Safe Zone Forecaster
  Stage 2: Expert Systems (Unit 4) - Forward-Chaining Rule-Based Macro-Strategy
  Stage 3: Autonomous Navigation & Heuristic Search (Units 1 & 2) - Threat-Aware A*

Luxury Brutalist Tactical Intelligence Suite
Palette: Dark Navy #08080c | Sage #b7c6c2 | Cyan #d5f4f9 | Taupe #9f8d8b | Charcoal #302b2f
"""

import os
import sys
import base64
import math
import time
from io import BytesIO
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Import core simulation engine
from zeus import *

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Z.E.U.S. • Zone Evaluation & Utility System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# Nodelight DeAI Design System & Modern Cybernetics Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400;1,600&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Hide standard sidebar and collapsed controls */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Hide Streamlit header, deploy button, and default menu */
    header[data-testid="stHeader"], [data-testid="stHeader"] {
        display: none !important;
    }
    #MainMenu, footer, .stDeployButton {
        display: none !important;
    }

    html {
        scroll-behavior: smooth;
    }

    /* Overall Nodelight Obsidian Canvas with Radiant Aura */
    html, body, .stApp {
        background-color: #08080c !important;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(124, 58, 237, 0.18) 0%, transparent 55%),
            radial-gradient(circle at 88% 22%, rgba(79, 70, 229, 0.12) 0%, transparent 45%),
            radial-gradient(circle at 12% 75%, rgba(139, 92, 246, 0.08) 0%, transparent 40%) !important;
        background-attachment: fixed !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        overflow-x: hidden;
    }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 7rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1650px !important;
        position: relative;
    }

    /* Smooth Breathing Ambient Aura Animations */
    @keyframes auraPulse {
        0% { transform: scale(1) translate(0px, 0px); opacity: 0.35; }
        50% { transform: scale(1.18) translate(30px, -20px); opacity: 0.55; }
        100% { transform: scale(1) translate(0px, 0px); opacity: 0.35; }
    }

    @keyframes auraSecondary {
        0% { transform: scale(1) translate(0px, 0px); opacity: 0.18; }
        50% { transform: scale(1.12) translate(-25px, 20px); opacity: 0.32; }
        100% { transform: scale(1) translate(0px, 0px); opacity: 0.18; }
    }

    .nl-ambient-orb-primary {
        position: absolute;
        width: 700px;
        height: 700px;
        top: -160px;
        left: calc(50% - 350px);
        border-radius: 50%;
        background: radial-gradient(circle, #7c3aed 0%, rgba(124, 58, 237, 0) 70%);
        filter: blur(140px);
        pointer-events: none;
        z-index: 0;
        animation: auraPulse 9s ease-in-out infinite;
    }

    .nl-ambient-orb-secondary {
        position: absolute;
        width: 520px;
        height: 520px;
        top: 200px;
        right: -80px;
        border-radius: 50%;
        background: radial-gradient(circle, #4f46e5 0%, rgba(79, 70, 229, 0) 70%);
        filter: blur(130px);
        pointer-events: none;
        z-index: 0;
        animation: auraSecondary 12s ease-in-out infinite;
    }

    /* Top Floating Navigation Bar */
    .nl-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.6rem;
        margin-bottom: 2.8rem;
        background: rgba(14, 14, 23, 0.72);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        position: relative;
        z-index: 10;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }

    .nl-nav-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nl-brand-logo {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nl-brand-icon {
        width: 32px;
        height: 32px;
        border-radius: 9px;
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 16px rgba(139, 92, 246, 0.55);
    }

    .nl-brand-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.28rem;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .nl-brand-badge {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #c4b5fd;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 2px 8px;
        border-radius: 9999px;
        font-family: 'JetBrains Mono', monospace;
    }

    .nl-nav-links {
        display: flex;
        align-items: center;
        gap: 22px;
    }

    .nl-nav-link {
        font-size: 0.88rem;
        font-weight: 500;
        color: #94a3b8;
        text-decoration: none;
        transition: all 0.25s ease;
    }

    .nl-nav-link:hover {
        color: #ffffff;
        text-shadow: 0 0 12px rgba(139, 92, 246, 0.7);
    }

    .nl-status-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .nl-status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulseDot 2s infinite ease-in-out;
    }

    @keyframes pulseDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    .nl-status-text {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6ee7b7;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Hero Section (Split Layout from Nodelight Template) */
    .nl-hero-container {
        display: grid;
        grid-template-columns: 1.15fr 0.85fr;
        gap: 3.5rem;
        align-items: center;
        margin-bottom: 3.5rem;
        position: relative;
        z-index: 2;
    }

    .nl-hero-left {
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .nl-tag-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #c4b5fd;
        margin-bottom: 1.2rem;
    }

    .nl-hero-title {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        font-size: clamp(3.2rem, 5.2vw, 4.6rem) !important;
        font-weight: 700 !important;
        line-height: 1.05 !important;
        letter-spacing: -0.035em !important;
        color: #ffffff !important;
        margin: 0 0 1.3rem 0 !important;
        text-transform: none !important;
    }

    .nl-hero-gradient {
        background: linear-gradient(135deg, #ffffff 30%, #c4b5fd 70%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nl-hero-desc {
        font-size: 1.05rem;
        line-height: 1.75;
        color: #94a3b8;
        max-width: 620px;
        font-weight: 400;
        margin: 0 0 2rem 0;
    }

    .nl-hero-cta-group {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .nl-btn-primary {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 0.78rem 1.9rem;
        border-radius: 9999px;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        border: 1px solid rgba(167, 139, 250, 0.4);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .nl-btn-primary:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6);
        border-color: #ffffff;
    }

    .nl-btn-secondary {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        padding: 0.78rem 1.8rem;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.04);
        color: #cbd5e1 !important;
        font-weight: 500;
        font-size: 0.95rem;
        text-decoration: none;
        border: 1px solid rgba(255, 255, 255, 0.14);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .nl-btn-secondary:hover {
        border-color: rgba(139, 92, 246, 0.5);
        background: rgba(139, 92, 246, 0.12);
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.2);
    }

    /* Hero Right Feature Card (matching "Catch momentum before it happens") */
    .nl-hero-feature-card {
        background: rgba(14, 14, 23, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }

    .nl-hero-feature-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.6), transparent);
    }

    .nl-hero-feature-card:hover {
        border-color: rgba(139, 92, 246, 0.45);
        transform: translateY(-4px);
        box-shadow: 0 20px 50px -10px rgba(124, 58, 237, 0.35);
    }

    .nl-feature-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.8rem;
    }

    .nl-feature-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        line-height: 1.2 !important;
        margin: 0 0 1rem 0 !important;
        letter-spacing: -0.02em !important;
    }

    .nl-feature-desc {
        font-size: 0.95rem;
        color: #94a3b8;
        line-height: 1.65;
        margin-bottom: 1.4rem;
    }

    .nl-feature-list {
        display: flex;
        flex-direction: column;
        gap: 11px;
        margin-bottom: 1.6rem;
    }

    .nl-feature-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        font-size: 0.88rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    .nl-feature-arrow {
        color: #8b5cf6;
        font-weight: 700;
        font-size: 1rem;
        line-height: 1.2;
    }

    .nl-feature-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        border-radius: 9999px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.3);
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #c4b5fd;
        font-weight: 600;
    }

    /* "What We Do" Section & 3-Card Grid (matching Nodelight template) */
    .nl-section-header {
        margin: 1.5rem 0 2rem 0;
    }

    .nl-section-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #a78bfa;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nl-section-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.025em !important;
        margin: 0 !important;
    }

    .nl-cards-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 24px;
        margin-bottom: 3.5rem;
    }

    .nl-card {
        background: rgba(14, 14, 22, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem 2rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        display: flex;
        flex-direction: column;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }

    .nl-card:hover {
        border-color: rgba(139, 92, 246, 0.45);
        transform: translateY(-6px);
        box-shadow: 0 16px 40px -10px rgba(124, 58, 237, 0.3);
        background: rgba(18, 18, 28, 0.85);
    }

    .nl-card-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.15em;
        color: #a78bfa;
        margin-bottom: 1.4rem;
    }

    .nl-card-icon {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    .nl-card:hover .nl-card-icon {
        background: rgba(139, 92, 246, 0.25);
        transform: scale(1.08);
        border-color: #a78bfa;
    }

    .nl-card-title {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.38rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 0 0 0.85rem 0 !important;
        letter-spacing: -0.015em !important;
        line-height: 1.3 !important;
    }

    .nl-card-desc {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.7;
        margin: 0 0 1.6rem 0;
        flex-grow: 1;
    }

    .nl-card-footer {
        display: flex;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #c4b5fd;
        font-weight: 600;
        letter-spacing: 0.08em;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Streamlit Components Redesign */
    div[data-testid="stMetric"] {
        background: rgba(14, 14, 23, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        padding: 1.4rem 1.6rem !important;
        backdrop-filter: blur(20px) !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
    }

    div[data-testid="stMetric"]:hover {
        border-color: rgba(139, 92, 246, 0.45) !important;
        transform: translateY(-4px) !important;
        box-shadow: 0 14px 35px -8px rgba(124, 58, 237, 0.3) !important;
        background: rgba(18, 18, 30, 0.85) !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 2.1rem !important;
        letter-spacing: -0.02em !important;
        color: #ffffff !important;
        line-height: 1.15 !important;
        margin-top: 0.2rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 11.5px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.15em !important;
        color: #a78bfa !important;
        margin-bottom: 0.35rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(14, 14, 23, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 20px !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 6px 28px rgba(0, 0, 0, 0.4) !important;
        padding: 1.8rem 2rem !important;
        margin-bottom: 1.8rem !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(139, 92, 246, 0.3) !important;
    }

    /* Buttons (Pill Shapes) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%) !important;
        border: 1px solid rgba(167, 139, 250, 0.4) !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 9999px !important;
        box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35) !important;
        padding: 0.65rem 1.6rem !important;
        margin: 0.3rem 0 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #9333ea 0%, #6d28d9 100%) !important;
        border-color: #ffffff !important;
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.55) !important;
    }

    button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        color: #cbd5e1 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        border-radius: 9999px !important;
        padding: 0.6rem 1.4rem !important;
        margin: 0.3rem 0 !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    button[kind="secondary"]:hover {
        border-color: rgba(139, 92, 246, 0.5) !important;
        color: #ffffff !important;
        background: rgba(139, 92, 246, 0.12) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.2) !important;
    }

    /* HUD Telemetry Cards */
    .hud-stat-box {
        background: rgba(14, 14, 23, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(16px);
        transition: all 0.3s ease;
    }

    .hud-stat-box:hover {
        border-color: rgba(139, 92, 246, 0.35);
        transform: translateY(-2px);
    }

    .hud-stat-label {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11.5px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #a78bfa;
        margin-bottom: 0.4rem;
    }

    .hud-val-blue {
        font-size: 1.65rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .hud-val-yellow {
        font-size: 1.65rem;
        font-weight: 700;
        color: #facc15;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .hud-val-green {
        font-size: 1.65rem;
        font-weight: 700;
        color: #34d399;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .hud-val-red {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f87171;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }

    .hud-sub-dim {
        font-size: 0.82rem;
        color: #94a3b8;
        font-weight: 500;
    }

    /* Badges */
    .hud-badge-blue {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        padding: 0.25rem 0.8rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    .hud-badge-yellow {
        display: inline-block;
        background: rgba(250, 204, 21, 0.12);
        color: #facc15;
        border: 1px solid rgba(250, 204, 21, 0.3);
        border-radius: 9999px;
        padding: 0.25rem 0.8rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    .hud-badge-green {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 9999px;
        padding: 0.25rem 0.8rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    .hud-badge-red {
        display: inline-block;
        background: rgba(239, 68, 68, 0.12);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 9999px;
        padding: 0.25rem 0.8rem;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
    }

    /* Sliders */
    div[data-testid="stSlider"] label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 12.5px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.12em !important;
        color: #c4b5fd !important;
        margin-bottom: 0.4rem !important;
    }

    /* Radio / Segmented Control */
    div[role="radiogroup"] label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        padding: 6px 14px !important;
        border-radius: 9999px !important;
        transition: all 0.25s ease !important;
    }

    /* Expanders (Matching FAQ Accordions from Nodelight Template) */
    div[data-testid="stExpander"] {
        background: rgba(12, 12, 19, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        margin-bottom: 1.2rem !important;
        backdrop-filter: blur(16px) !important;
        overflow: hidden !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    div[data-testid="stExpander"]:hover {
        border-color: rgba(139, 92, 246, 0.38) !important;
        box-shadow: 0 8px 25px -6px rgba(124, 58, 237, 0.25) !important;
    }

    div[data-testid="stExpander"] summary {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        color: #ffffff !important;
        padding: 1.2rem 1.6rem !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stExpander"] summary:hover {
        color: #c4b5fd !important;
        background: rgba(139, 92, 246, 0.08) !important;
    }

    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 1.4rem 1.8rem 1.8rem 1.8rem !important;
        border-top: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(8, 8, 14, 0.5) !important;
    }


    /* Big Glassmorphic Title Bar */
    
    /* Big Glassmorphic Title Bar */
    
    /* Sleek & Compact Glassmorphic Title Bar */
    .nl-titlebar-glass {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.85rem 1.8rem;
        margin-bottom: 2.2rem;
        background: rgba(14, 14, 24, 0.78);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.5), 0 0 24px rgba(124, 58, 237, 0.18);
        position: relative;
        z-index: 10;
        transition: all 0.35s ease;
    }

    .nl-titlebar-glass:hover {
        border-color: rgba(139, 92, 246, 0.45);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.6), 0 0 30px rgba(124, 58, 237, 0.28);
    }

    .nl-titlebar-content {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .nl-titlebar-logo-img {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        object-fit: cover;
        box-shadow: 0 0 16px rgba(168, 85, 247, 0.45);
        border: 1px solid rgba(168, 85, 247, 0.35);
        display: block;
    }

    .nl-titlebar-main {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.06em !important;
        color: #ffffff !important;
        line-height: 1 !important;
        text-shadow: 0 0 22px rgba(139, 92, 246, 0.55) !important;
        margin: 0 !important;
    }

    .nl-fullform-badge {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: #c4b5fd !important;
        padding: 5px 13px !important;
        border-radius: 9999px !important;
        background: rgba(139, 92, 246, 0.12) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        display: inline-flex !important;
        align-items: center !important;
        backdrop-filter: blur(12px) !important;
        margin-left: 6px !important;
    }

    .nl-header-right-group {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .nl-nav-pill-link {
        display: inline-flex;
        align-items: center;
        padding: 7px 16px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #cbd5e1 !important;
        text-decoration: none !important;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.04em;
        transition: all 0.25s ease;
        backdrop-filter: blur(16px);
    }

    .nl-nav-pill-link:hover {
        background: rgba(139, 92, 246, 0.18);
        border-color: rgba(139, 92, 246, 0.5);
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        transform: translateY(-2px);
    }

    /* Centered Hero Section (Pro-portioned for Image 2) */
    .nl-hero-center-box {
        text-align: center;
        max-width: 1080px;
        margin: 2rem auto 3rem auto;
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Background Watermark & Tactical Rings */
    .nl-hero-map-backdrop {
        position: absolute;
        top: 48%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 960px;
        height: 620px;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.13;
        filter: grayscale(100%) contrast(150%) brightness(85%);
        mask-image: radial-gradient(ellipse 65% 65% at 50% 50%, #000 20%, transparent 80%);
        -webkit-mask-image: radial-gradient(ellipse 65% 65% at 50% 50%, #000 20%, transparent 80%);
        pointer-events: none;
        z-index: 0;
    }

    .nl-hero-tactical-rings {
        position: absolute;
        top: 48%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 760px;
        height: 760px;
        border: 1px dashed rgba(139, 92, 246, 0.22);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }

    .nl-hero-tactical-rings::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 480px;
        height: 480px;
        border: 1px solid rgba(56, 189, 248, 0.16);
        border-radius: 50%;
    }

    .nl-hero-tactical-rings::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 240px;
        height: 240px;
        border: 1px solid rgba(234, 179, 8, 0.16);
        border-radius: 50%;
    }

    .nl-tag-pill-large {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: #c4b5fd;
        margin-bottom: 1.8rem;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.35);
        padding: 7px 22px;
        border-radius: 9999px;
        position: relative;
        z-index: 1;
    }

    .nl-hero-title-center {
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
        font-size: clamp(4.0rem, 7.2vw, 6.0rem) !important;
        font-weight: 800 !important;
        line-height: 1.08 !important;
        letter-spacing: -0.035em !important;
        color: #ffffff !important;
        margin: 0 0 1.8rem 0 !important;
        text-transform: none !important;
        position: relative;
        z-index: 1;
    }

    .nl-hero-desc-center {
        font-size: 1.22rem !important;
        line-height: 1.78 !important;
        color: #cbd5e1 !important;
        font-weight: 400 !important;
        max-width: 820px !important;
        text-align: center !important;
        margin: 0 auto 2.5rem auto !important;
        position: relative;
        z-index: 1;
    }

    /* Live Metric Strip (4-Column) */
    .nl-metric-strip-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 18px;
        max-width: 1020px;
        margin: 3.2rem auto 1.8rem auto;
        position: relative;
        z-index: 2;
    }

    .nl-strip-card {
        background: rgba(14, 14, 24, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        backdrop-filter: blur(20px);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }

    .nl-strip-card:hover {
        border-color: rgba(139, 92, 246, 0.45);
        transform: translateY(-4px);
        box-shadow: 0 14px 32px rgba(124, 58, 237, 0.25);
        background: rgba(18, 18, 30, 0.85);
    }

    .nl-strip-val {
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        line-height: 1.1 !important;
        margin-bottom: 0.3rem !important;
    }

    .nl-strip-label {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.14em !important;
        color: #a78bfa !important;
        margin-bottom: 0.2rem !important;
    }

    .nl-strip-sub {
        font-size: 0.8rem !important;
        color: #94a3b8 !important;
    }

    /* Bottom Tech Pills */
    .nl-tech-pills-row {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 1.2rem;
        z-index: 2;
        position: relative;
    }

    .nl-tech-pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: rgba(255, 255, 255, 0.03);
        border-radius: 9999px;
        padding: 7px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        color: #cbd5e1;
        font-weight: 500;
        letter-spacing: 0.03em;
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }

    .nl-tech-pill-badge:hover {
        border-color: rgba(139, 92, 246, 0.4);
        background: rgba(139, 92, 246, 0.1);
        color: #ffffff;
        transform: translateY(-2px);
    }

    .nl-pill-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }

    /* Button Styling - Fix for Image 1 (NO UNDERLINE AT ALL) */
    div[data-testid="stButton"] button {
        text-decoration: none !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stButton"] button p {
        text-decoration: none !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
    }
    div[data-testid="stButton"] button:hover {
        text-decoration: none !important;
    }
    div[data-testid="stButton"] button:focus {
        text-decoration: none !important;
    }

    /* Terminal Console */
    .cyber-terminal {
        background: #09090f;
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 12px;
        padding: 1.4rem 1.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #c4b5fd;
        line-height: 1.75;
        max-height: 360px;
        overflow-y: auto;
    }

    .cyber-terminal-header {
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid rgba(139, 92, 246, 0.2);
        padding-bottom: 0.65rem;
        margin-bottom: 1rem;
        font-size: 0.8rem;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-weight: 700;
    }

    /* Smooth Scrolling for the Whole Single Page */
    html, body {
        scroll-behavior: smooth !important;
    }

    /* "Go to the Map" Action Button - Centered, Vibrant Glow, NO Underline */
    .nl-btn-go-map-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.8rem 0 2.4rem 0;
        position: relative;
        z-index: 10;
    }

    a.nl-btn-go-map, .nl-btn-go-map {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 12px !important;
        background: linear-gradient(135deg, #9333ea 0%, #7c3aed 50%, #4f46e5 100%) !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.18rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.95rem 2.8rem !important;
        border-radius: 9999px !important;
        text-decoration: none !important;
        border-bottom: none !important;
        box-shadow: 0 0 28px rgba(147, 51, 234, 0.55), 0 8px 24px rgba(0, 0, 0, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        cursor: pointer !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    a.nl-btn-go-map:hover, .nl-btn-go-map:hover {
        background: linear-gradient(135deg, #a855f7 0%, #8b5cf6 50%, #6366f1 100%) !important;
        color: #ffffff !important;
        text-decoration: none !important;
        border-bottom: none !important;
        transform: translateY(-3px) scale(1.03) !important;
        box-shadow: 0 0 40px rgba(168, 85, 247, 0.75), 0 14px 32px rgba(0, 0, 0, 0.6) !important;
    }

    a.nl-btn-go-map:active, a.nl-btn-go-map:focus, a.nl-btn-go-map:visited,
    .nl-btn-go-map:active, .nl-btn-go-map:focus, .nl-btn-go-map:visited {
        color: #ffffff !important;
        text-decoration: none !important;
        border-bottom: none !important;
        outline: none !important;
    }

    .nl-btn-go-map span, .nl-btn-go-map * {
        text-decoration: none !important;
        border-bottom: none !important;
    }

    .bounce-arrow {
        display: inline-block;
        font-size: 1.25rem;
        font-weight: 800;
        animation: bounceArrowAnim 1.6s infinite ease-in-out;
    }

    @keyframes bounceArrowAnim {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(6px);
        }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Ambient Background Orbs & Nodelight DeAI Executive Header
# -----------------------------------------------------------------------------


# Map Asset & Image Preparation
# -----------------------------------------------------------------------------
MAP_FILE = "Erangel_Main_Low_Res.png"
if not os.path.exists(MAP_FILE):
    st.error(f"Required map file {MAP_FILE} missing.")
    st.stop()

raw_erangel_img = Image.open(MAP_FILE).convert("RGB")
IMG_W, IMG_H = raw_erangel_img.size

@st.cache_data
def get_erangel_base64_uri():
    buffered = BytesIO()
    raw_erangel_img.save(buffered, format="JPEG", quality=85)
    b64_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64_str}"

ERANGEL_URI = get_erangel_base64_uri()
@st.cache_data
def get_logo_base64_uri():
    candidates = [
        ("Z.E.U.S/logo.png", "image/png"),
        ("logo.png", "image/png"),
        ("Z.E.U.S/logo.webp", "image/webp"),
        ("logo.webp", "image/webp"),
        ("../Z.E.U.S/logo.png", "image/png"),
        ("../logo.png", "image/png"),
    ]
    for p, mime in candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return ""

LOGO_URI = get_logo_base64_uri()


# Canonical Landmark Locations (Normalized [0, 1])
LANDMARKS_NORM = {
    "Georgopol": (0.22, 0.31),
    "Gatka": (0.27, 0.48),
    "Pochinki": (0.45, 0.49),
    "School": (0.52, 0.41),
    "Rozhok": (0.50, 0.35),
    "Yasnaya Polyana": (0.67, 0.29),
    "Mylta": (0.76, 0.57),
    "West Bridge": (0.42, 0.69),
    "East Bridge": (0.67, 0.70),
    "Sosnovka Military Base": (0.54, 0.81),
    "Novorepnoye": (0.76, 0.75),
    "Severny": (0.46, 0.15),
    "Stalber": (0.71, 0.16)
}

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "app_view" not in st.session_state:
    st.session_state["app_view"] = "landing"

if "workflow_step" not in st.session_state:
    st.session_state["workflow_step"] = 1  # 1: Zone -> 2: Squad -> 3: Enemies -> 4: Simulate

if "squad_pos" not in st.session_state:
    st.session_state["squad_pos"] = LANDMARKS_NORM["Gatka"]

if "zone_center" not in st.session_state:
    st.session_state["zone_center"] = LANDMARKS_NORM["Pochinki"]

if "zone_radius_pct" not in st.session_state:
    st.session_state["zone_radius_pct"] = 0.32

if "next_zone_center" not in st.session_state:
    st.session_state["next_zone_center"] = (0.50, 0.45)

if "next_zone_radius_pct" not in st.session_state:
    st.session_state["next_zone_radius_pct"] = 0.18

if "zone_edit_target" not in st.session_state:
    st.session_state["zone_edit_target"] = "blue"

if "next_zone_manual" not in st.session_state:
    st.session_state["next_zone_manual"] = False

if "campers" not in st.session_state:
    st.session_state["campers"] = []

if "strategy_mode" not in st.session_state:
    st.session_state["strategy_mode"] = "Auto (IGL)"

if "last_click" not in st.session_state:
    st.session_state["last_click"] = None

if "time_left" not in st.session_state:
    st.session_state["time_left"] = 45

if "map_layer" not in st.session_state:
    st.session_state["map_layer"] = "Satellite (Erangel)"

if "active_topology" not in st.session_state:
    st.session_state["active_topology"] = "erangel"

if "anim_step" not in st.session_state:
    st.session_state["anim_step"] = 0

# -----------------------------------------------------------------------------
# Discretize Erangel Terrain (Cover & Obstacles)
# -----------------------------------------------------------------------------
N_GRID = 80

@st.cache_data
def process_erangel_grid(n):
    resized = raw_erangel_img.resize((n, n), Image.Resampling.BILINEAR)
    arr = np.array(resized).astype(float) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    yy, xx = np.indices((n, n))
    dist = np.hypot(yy - n * 0.48, xx - n * 0.48)

    # Water detection
    is_water = (b > r * 1.35) & (b > 0.12)
    is_water |= (dist > 0.46 * n)

    # Preserve Sosnovka Military Bridges
    wb_y, wb_x = int(0.69 * n), int(0.42 * n)
    eb_y, eb_x = int(0.70 * n), int(0.67 * n)
    bw = max(1, int(0.025 * n))
    is_water[wb_y - bw:wb_y + bw + 1, wb_x - bw:wb_x + bw + 1] = False
    is_water[eb_y - bw:eb_y + bw + 1, eb_x - bw:eb_x + bw + 1] = False

    blocked = is_water
    brightness = 0.299 * r + 0.587 * g + 0.114 * b
    cover = np.clip((brightness - 0.25) * 1.8, 0.0, 0.85)
    cover[blocked] = 0.0
    return cover, blocked

erangel_cover, erangel_blocked = process_erangel_grid(N_GRID)

# -----------------------------------------------------------------------------
# Core Engine Calculations for Active Setup
# -----------------------------------------------------------------------------
time_left = st.session_state.get("time_left", 45)
cur_squad = st.session_state["squad_pos"]
cur_center = st.session_state["zone_center"]
cur_radius = st.session_state["zone_radius_pct"]
cur_next_center = st.session_state.get("next_zone_center", (0.50, 0.45))
cur_next_radius = st.session_state.get("next_zone_radius_pct", 0.18)

camper_coords = []
for cx, cy in st.session_state["campers"]:
    gy, gx = int(cy * N_GRID), int(cx * N_GRID)
    if 0 <= gy < N_GRID and 0 <= gx < N_GRID:
        camper_coords.append((gy, gx))

rng = np.random.default_rng(42)
threat, enemies = make_threat(erangel_blocked, rng, k=0, extra=camper_coords)

c_center_grid = (int(cur_center[1] * N_GRID), int(cur_center[0] * N_GRID))
r0_grid = max(3, int(cur_radius * N_GRID))
r_next_grid = max(2, int(cur_next_radius * N_GRID))
next_center_grid = (int(cur_next_center[1] * N_GRID), int(cur_next_center[0] * N_GRID))
next_c_arg = next_center_grid if st.session_state.get("next_zone_manual", False) else None

target_start_grid = (int(cur_squad[1] * N_GRID), int(cur_squad[0] * N_GRID))
pos = nearest_free(~erangel_blocked, target_start_grid)

strat_map = {
    "Auto (IGL)": "AUTO",
    "Fortified Compound": "COMPOUND",
    "Open Ridge Bunker": "RIDGE_CENTER",
    "Edge Gatekeep": "GATEKEEP"
}
internal_strat = strat_map.get(st.session_state.get("strategy_mode", "Auto (IGL)"), "AUTO")

# Run Complete IGL Decision Pipeline
res = igl_decision(
    cover=erangel_cover,
    blocked=erangel_blocked,
    threat=threat,
    pos=pos,
    zone_c=c_center_grid,
    r_now=r0_grid,
    r_next=r_next_grid,
    time_left=time_left,
    history_d2=[40, 90],
    strategy_mode=internal_strat,
    enemies=enemies,
    next_c=next_c_arg
)

if not st.session_state.get("next_zone_manual", False):
    st.session_state["next_zone_center"] = (res["goal"][1] / N_GRID, res["goal"][0] / N_GRID)
    cur_next_center = st.session_state["next_zone_center"]

goal = res["goal"]
target_pos = res.get("target_pos", goal)
active_strategy = res.get("active_strategy", "AUTO")
target_terrain = res.get("target_terrain", "Fortified Compound")
decision = res["decision"]
facts = res["facts"]
rules_fired = res["rules_fired"]
in_zone = res.get("in_zone", False)
in_next_zone = res.get("in_next_zone", False)
enemies_outside_next = res.get("enemies_outside_next", [])
enemies_inside_next = res.get("enemies_inside_next", [])
gatekeep_info = res.get("gatekeep_info", {})
tactical_advice = res.get("tactical_advice", "")
protocols = res.get("protocols", [])
risks = res.get("risks", [])

base_rt = res["baseline_route"]
threat_rt = res["route"]

# Telemetry Calculations
base_surv = survival(base_rt["path"], threat, erangel_cover) if (base_rt and base_rt.get("found", False)) else (1.0 if not enemies else 0.0)
threat_surv = survival(threat_rt["path"], threat, erangel_cover) if (threat_rt and threat_rt.get("found", False)) else (1.0 if not enemies else 0.0)

def calc_threat_exposure(path, threat_map):
    if not path:
        return 0.0
    return sum(threat_map[y, x] for y, x in path)

base_exp = calc_threat_exposure(base_rt.get("path", []), threat)
threat_exp = calc_threat_exposure(threat_rt.get("path", []), threat)

scale_x = IMG_W / N_GRID
scale_y = IMG_H / N_GRID

path_pts = threat_rt.get("path", []) if (threat_rt and threat_rt.get("found", False)) else []
num_steps = max(0, len(path_pts) - 1) if path_pts else 0
distance_m = int(num_steps * (8000 / N_GRID))
vehicle_speed_kmh = 80.0
foot_speed_kmh = 22.0
vehicle_eta_sec = int((distance_m / 1000.0) / vehicle_speed_kmh * 3600) + (4 if distance_m > 0 else 0)
foot_eta_sec = int((distance_m / 1000.0) / foot_speed_kmh * 3600)

waypoints = []
if len(path_pts) >= 4:
    idx_list = np.linspace(0, len(path_pts) - 1, 4, dtype=int)
    for i, idx in enumerate(idx_list):
        pt_y, pt_x = path_pts[idx]
        waypoints.append((
            f"WP {i+1}" if (i > 0 and i < 3) else ("SPAWN" if i == 0 else "TARGET"),
            (pt_x * scale_x, pt_y * scale_y)
        ))

min_camper_dist_m = 9999
if enemies and path_pts:
    for ey, ex in enemies:
        for pt_y, pt_x in path_pts:
            d = math.hypot(pt_x - ex, pt_y - ey) * (8000 / N_GRID)
            if d < min_camper_dist_m:
                min_camper_dist_m = int(d)

dist_to_center_m = int(math.dist(pos, c_center_grid) * (8000 / N_GRID))
dist_to_edge_m = int(abs(math.dist(pos, c_center_grid) - r0_grid) * (8000 / N_GRID))
dist_to_yellow_m = int(abs(math.dist(pos, goal) - r_next_grid) * (8000 / N_GRID))
zone_diam_m = int(cur_radius * 2 * 8000)
next_zone_diam_m = int(cur_next_radius * 2 * 8000)

# Helper function for gatekeeping field of fire sector cone
def generate_cone_pts(x0, y0, r_px, b_start, b_end, num_pts=30):
    sweep = (b_end - b_start) % 360
    if sweep == 0:
        sweep = 360
    bearings = [b_start + sweep * (i / (num_pts - 1)) for i in range(num_pts)]
    xs, ys = [x0], [y0]
    for b in bearings:
        rad = math.radians(b)
        xs.append(x0 + r_px * math.sin(rad))
        ys.append(y0 - r_px * math.cos(rad))
    xs.append(x0)
    ys.append(y0)
    return xs, ys


# =============================================================================
# SINGLE-PAGE TACTICAL DASHBOARD: HERO SECTION -> RADAR HUD COMMAND CENTER
# =============================================================================

# Ambient background glows
st.markdown("""<div class="nl-ambient-orb-primary"></div>
<div class="nl-ambient-orb-secondary"></div>""", unsafe_allow_html=True)

# 1. Sleek Glassmorphic Title Bar with Logo & Full Form Next to ZEUS
st.markdown(f"""<div class="nl-titlebar-glass">
<div class="nl-titlebar-content">
<img src="{LOGO_URI}" class="nl-titlebar-logo-img" alt="Z.E.U.S. Logo" />
<span class="nl-titlebar-main">Z.E.U.S.</span>
<div class="nl-fullform-badge">ZONE EVALUATION AND UTILITY SYSTEM</div>
</div>
<div class="nl-header-right-group">
<a href="https://github.com/SUJALPATEL15/zeus" target="_blank" class="nl-nav-pill-link">
<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style="vertical-align: -2px; margin-right: 5px;">
<path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
</svg>
GitHub
</a>
<div class="nl-status-pill">
<span class="nl-status-dot"></span>
<span class="nl-status-text">ESPORTS IGL WORKBENCH</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

# 2. Centered Hero Section with Watermark & Tactical Rings (FLUSH LEFT - ZERO INDENTATION)
st.markdown(f"""<div class="nl-hero-center-box">
<div class="nl-hero-map-backdrop" style="background-image: url('{ERANGEL_URI}');"></div>
<div class="nl-hero-tactical-rings"></div>
<div class="nl-tag-pill-large">| PRO SCRIMMAGE WORKBENCH // TOURNAMENT INTELLIGENCE |</div>
<h1 class="nl-hero-title-center">
Predict the Shift.<br>
<span class="nl-hero-gradient">Perfect the Rotation.</span>
</h1>
<p class="nl-hero-desc-center">
Designed as a high-fidelity simulation and scrimmage workbench for <b style="color: #ffffff;">professional IGLs, coaches, and competitive squads</b>. Z.E.U.S. lets your roster <span style="color: #c4b5fd; font-weight: 600;">dry-run late-game circle shifts</span>, analyze <span style="color: #c4b5fd; font-weight: 600;">high-risk choke points</span>, and <span style="color: #c4b5fd; font-weight: 600;">stress-test pathing options</span> across hundreds of tournament-grade ring collapses before stepping onto the stage.
</p>
</div>""", unsafe_allow_html=True)

# 4. 4-Column Metric Strip to Balance the Lower Hero Area
st.markdown("""<div class="nl-metric-strip-grid">
<div class="nl-strip-card">
<div class="nl-strip-val">0.0s</div>
<div class="nl-strip-label">Blue Damage</div>
<div class="nl-strip-sub">Zero Incurred Exposure</div>
</div>
<div class="nl-strip-card">
<div class="nl-strip-val">8-Phase</div>
<div class="nl-strip-label">Shrink Support</div>
<div class="nl-strip-sub">P1 Landmass to Final Ring</div>
</div>
<div class="nl-strip-card">
<div class="nl-strip-val">Erangel & Miramar</div>
<div class="nl-strip-label">Topology Grids</div>
<div class="nl-strip-sub">Compound & Ridge Formations</div>
</div>
<div class="nl-strip-card">
<div class="nl-strip-val">&lt; 15ms</div>
<div class="nl-strip-label">Search Latency</div>
<div class="nl-strip-sub">Real-Time Threat-A* Planning</div>
</div>
</div>""", unsafe_allow_html=True)

# 6. Smooth Scroll Anchor & Section Header directly below the 1st dashboard
st.markdown("""<div id="erangel-radar-map" style="scroll-margin-top: 30px; margin-top: 3.5rem; margin-bottom: 1.5rem;"></div>
<div id="radar-hud"></div>
<div class="nl-section-header" style="margin-top: 1rem; margin-bottom: 1.2rem;">
<div class="nl-section-tag">⚡ Interactive Command Center</div>
<h2 class="nl-section-title">Erangel Tactical Radar & Navigation HUD</h2>
<p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem;">
Tournament simulation sandbox: predict safe zone shifts, deploy squads, detect hostiles, and execute threat-aware pathfinding.
</p>
</div>""", unsafe_allow_html=True)

current_step = st.session_state["workflow_step"]

step_c1, step_c2, step_c3, step_c4 = st.columns(4)
if step_c1.button("🎯 Phase 01: Dual Zones", key="wf_step_1", use_container_width=True, type="primary" if current_step == 1 else "secondary"):
    st.session_state["workflow_step"] = 1
    st.rerun()
if step_c2.button("📍 Phase 02: Squad Deploy", key="wf_step_2", use_container_width=True, type="primary" if current_step == 2 else "secondary"):
    st.session_state["workflow_step"] = 2
    st.rerun()
if step_c3.button("⚠️ Phase 03: Hostile Recon", key="wf_step_3", use_container_width=True, type="primary" if current_step == 3 else "secondary"):
    st.session_state["workflow_step"] = 3
    st.rerun()
if step_c4.button("⚡ Phase 04: IGL Simulation", key="wf_step_4", use_container_width=True, type="primary" if current_step == 4 else "secondary"):
    st.session_state["workflow_step"] = 4
    st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Top KPI Metrics Strip
top_m1, top_m2, top_m3, top_m4 = st.columns(4)

if current_step < 4:
    top_m1.metric("Current Zone 🔵", f"{zone_diam_m}m", f"Radius {int(cur_radius*100)}% map")
    top_m2.metric("Next Zone 🟡", f"{next_zone_diam_m}m", f"Radius {int(cur_next_radius*100)}% map")
    if in_next_zone:
        squad_stat, squad_sub, delta_c = "Inside Next Zone 🟡", "Prime Defensive Position", "normal"
    elif in_zone:
        squad_stat, squad_sub, delta_c = "Inside Blue Zone 🔵", f"{dist_to_yellow_m}m to Yellow line", "off"
    else:
        squad_stat, squad_sub, delta_c = "Outside Blue Zone 🔴", f"{dist_to_edge_m}m to Blue line", "inverse"
    top_m3.metric("Squad Status", squad_stat, squad_sub, delta_color=delta_c)
    top_m4.metric("Spotted Enemies", f"{len(enemies)} Squads", "Clean Sector" if not enemies else f"{len(enemies_outside_next)} out • {len(enemies_inside_next)} in")
else:
    if in_next_zone:
        squad_stat, squad_delta, delta_c = "Inside Next Zone 🟡", "Safe Zone Secured", "normal"
    elif in_zone:
        squad_stat, squad_delta, delta_c = "Inside Blue Zone 🔵", f"{dist_to_yellow_m}m to Yellow line", "off"
    else:
        squad_stat, squad_delta, delta_c = "Outside Blue Zone 🔴", f"{dist_to_edge_m}m to Blue line", "inverse"

    top_m1.metric("Squad Position", squad_stat, squad_delta, delta_color=delta_c)
    top_m2.metric(
        "Collapse Timer",
        f"{time_left}s",
        f"{time_left - vehicle_eta_sec:+d}s vehicle margin",
        delta_color="normal" if time_left >= vehicle_eta_sec else "inverse"
    )
    top_m3.metric(
        "Hostile Breakdown",
        f"{len(enemies)} Squads",
        f"{len(enemies_outside_next)} Outside • {len(enemies_inside_next)} Inside" if enemies else "Clean Sector"
    )
    if not in_next_zone:
        top_m4.metric("Priority Directive", "ROTATE TO YELLOW", f"Target: {target_terrain}")
    elif active_strategy == "GATEKEEP ZONE PERIMETER":
        top_m4.metric("Priority Directive", "GATEKEEP OUTSIDE", f"Aim: {int(gatekeep_info.get('mean_bearing', 0))}° {gatekeep_info.get('primary_sector', '').upper()}")
    else:
        top_m4.metric("Priority Directive", "HOLD SECURE", f"Yellow Zone: {target_terrain}")

st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

# Radar HUD Bar & Controls
hud_c1, hud_c2, hud_c3 = st.columns([1.5, 1.2, 1.3])
with hud_c1:
    map_layer = st.radio(
        "HUD Sensor Layer",
        ["Satellite (Erangel)", "🔥 Threat Heatmap", "🔮 Bayesian Forecast", "🛡️ Cover Topography"],
        horizontal=True,
        key="radio_map_layer"
    )
with hud_c2:
    if current_step == 4:
        selected_strat = st.segmented_control(
            "Strategic Objective",
            options=["Auto (IGL)", "Fortified Compound", "Open Ridge Bunker", "Edge Gatekeep"],
            default=st.session_state["strategy_mode"]
        )
        if selected_strat and selected_strat != st.session_state["strategy_mode"]:
            st.session_state["strategy_mode"] = selected_strat
            st.rerun()
with hud_c3:
    if current_step == 4 and path_pts and len(path_pts) > 1:
        anim_idx = st.slider("Tactical Route Playback", 0, len(path_pts) - 1, int(st.session_state.get("anim_step", 0)), key="sim_anim_slider")
        st.session_state["anim_step"] = anim_idx
    else:
        st.session_state["anim_step"] = 0

st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

col_map, col_ops = st.columns([1.45, 1.0], gap="large")

with col_map:
    fig = go.Figure()

    # Render Selected Layer
    if map_layer == "🔥 Threat Heatmap":
        fig.add_trace(go.Heatmap(
            z=threat, x=np.linspace(0, IMG_W, N_GRID), y=np.linspace(0, IMG_H, N_GRID),
            colorscale="Reds", showscale=False, opacity=0.88, hoverinfo="none"
        ))
    elif map_layer == "🔮 Bayesian Forecast":
        fc_obj = ZoneForecaster(erangel_cover, erangel_blocked)
        post_grid = fc_obj.posterior(c_center_grid, r0_grid, r_next_grid, [40, 90])
        fig.add_trace(go.Heatmap(
            z=post_grid, x=np.linspace(0, IMG_W, post_grid.shape[1]), y=np.linspace(0, IMG_H, post_grid.shape[0]),
            colorscale="Viridis", showscale=False, opacity=0.85, hoverinfo="none"
        ))
    elif map_layer == "🛡️ Cover Topography":
        fig.add_trace(go.Heatmap(
            z=erangel_cover, x=np.linspace(0, IMG_W, N_GRID), y=np.linspace(0, IMG_H, N_GRID),
            colorscale="Greens", showscale=False, opacity=0.85, hoverinfo="none"
        ))
    else:
        # Satellite imagery background
        fig.add_layout_image(
            dict(
                source=ERANGEL_URI,
                xref="x", yref="y",
                x=0, y=0,
                sizex=IMG_W, sizey=IMG_H,
                sizing="stretch",
                layer="below"
            )
        )

    fig.update_xaxes(range=[0, IMG_W], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[IMG_H, 0], showgrid=False, zeroline=False, visible=False)

    # Clickable Hit Grid
    gx, gy = np.meshgrid(np.linspace(15, IMG_W - 15, 42), np.linspace(15, IMG_H - 15, 42))
    if current_step == 1:
        hover_prompts = [f"Click to set Playzone Center at ({int(x)}, {int(y)})" for x, y in zip(gx.flatten(), gy.flatten())]
    elif current_step == 2:
        hover_prompts = [f"Click to place Squad at ({int(x)}, {int(y)})" for x, y in zip(gx.flatten(), gy.flatten())]
    elif current_step == 3:
        hover_prompts = [f"Click to place Hostile Squad at ({int(x)}, {int(y)})" for x, y in zip(gx.flatten(), gy.flatten())]
    else:
        hover_prompts = [f"Click to reposition Squad to ({int(x)}, {int(y)})" for x, y in zip(gx.flatten(), gy.flatten())]

    fig.add_trace(go.Scatter(
        x=gx.flatten(), y=gy.flatten(),
        mode="markers",
        marker=dict(size=18, opacity=0.001, color="#ffffff"),
        hoverinfo="text",
        hovertext=hover_prompts,
        name="Click Canvas"
    ))

    theta = np.linspace(0, 2 * np.pi, 120)

    # 1. Blue Playzone Circle
    circ_x = cur_center[0] * IMG_W
    circ_y = cur_center[1] * IMG_H
    circ_r = cur_radius * IMG_W
    wx = circ_x + circ_r * np.cos(theta)
    wy = circ_y + circ_r * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=wx, y=wy, mode="lines",
        line=dict(color="#38bdf8", width=3.0),
        name="Current Zone (Blue)",
        hoverinfo="skip"
    ))

    is_blue_active = (current_step == 1 and st.session_state.get("zone_edit_target", "blue") == "blue")
    fig.add_trace(go.Scatter(
        x=[circ_x], y=[circ_y],
        mode="markers+text",
        marker=dict(
            size=12 if is_blue_active else 8,
            color="#38bdf8",
            symbol="cross",
            line=dict(color="#ffffff", width=2.0 if is_blue_active else 1.0)
        ),
        text=["BLUE CENTER"] if current_step == 1 else [""],
        textposition="bottom center",
        textfont=dict(color="#38bdf8", size=9),
        name="Blue Center",
        hoverinfo="skip"
    ))

    # 2. Yellow Safe Zone Circle
    next_px_x = cur_next_center[0] * IMG_W
    next_px_y = cur_next_center[1] * IMG_H
    next_px_r = cur_next_radius * IMG_W
    gx_pts = next_px_x + next_px_r * np.cos(theta)
    gy_pts = next_px_y + next_px_r * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=gx_pts, y=gy_pts, mode="lines",
        line=dict(color="#eab308", width=2.5, dash="dash"),
        name="Next Zone (Yellow)",
        hoverinfo="skip"
    ))

    is_yellow_active = (current_step == 1 and st.session_state.get("zone_edit_target", "blue") == "yellow")
    fig.add_trace(go.Scatter(
        x=[next_px_x], y=[next_px_y],
        mode="markers+text",
        marker=dict(
            size=12 if is_yellow_active else 8,
            color="#eab308",
            symbol="cross",
            line=dict(color="#ffffff", width=2.0 if is_yellow_active else 1.0)
        ),
        text=["YELLOW CENTER"] if current_step == 1 else [""],
        textposition="top center",
        textfont=dict(color="#eab308", size=9),
        name="Yellow Center",
        hoverinfo="skip"
    ))

    # 3. Squad Marker
    start_px_x = pos[1] * scale_x
    start_px_y = pos[0] * scale_y
    if current_step >= 2:
        fig.add_trace(go.Scatter(
            x=[start_px_x], y=[start_px_y],
            mode="markers+text",
            marker=dict(size=14, color="#0284c7", symbol="circle", line=dict(color="#ffffff", width=2.0)),
            text=["SQUAD"], textposition="top center",
            textfont=dict(color="#38bdf8", size=10, family="sans-serif"),
            name="Squad Location"
        ))

    # 4. Hostile Camper Squads
    if current_step == 3 and enemies:
        camper_r_px = (150.0 / 8000.0) * IMG_W
        for ey, ex in enemies:
            cx_ring = ex * scale_x + camper_r_px * np.cos(theta)
            cy_ring = ey * scale_y + camper_r_px * np.sin(theta)
            fig.add_trace(go.Scatter(
                x=cx_ring, y=cy_ring, mode="lines",
                line=dict(color="rgba(239, 68, 68, 0.45)", width=1.2, dash="dot"),
                hoverinfo="skip", showlegend=False
            ))
        ex_pts = [e[1] * scale_x for e in enemies]
        ey_pts = [e[0] * scale_y for e in enemies]
        fig.add_trace(go.Scatter(
            x=ex_pts, y=ey_pts,
            mode="markers",
            marker=dict(size=11, color="#ef4444", symbol="x", line=dict(color="#ffffff", width=1.5)),
            name=f"Campers ({len(enemies)})",
            hovertemplate="Enemy Camper Squad<extra></extra>"
        ))
    elif current_step == 4 and enemies:
        camper_r_px = (150.0 / 8000.0) * IMG_W
        if enemies_outside_next:
            for ey, ex in enemies_outside_next:
                cx_ring = ex * scale_x + camper_r_px * np.cos(theta)
                cy_ring = ey * scale_y + camper_r_px * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=cx_ring, y=cy_ring, mode="lines",
                    line=dict(color="rgba(239, 68, 68, 0.50)", width=1.4, dash="dot"),
                    hoverinfo="skip", showlegend=False
                ))
            ex_pts_out = [e[1] * scale_x for e in enemies_outside_next]
            ey_pts_out = [e[0] * scale_y for e in enemies_outside_next]
            fig.add_trace(go.Scatter(
                x=ex_pts_out, y=ey_pts_out,
                mode="markers",
                marker=dict(size=12, color="#ef4444", symbol="x", line=dict(color="#ffffff", width=1.5)),
                name=f"Outside Yellow ({len(enemies_outside_next)})",
                hovertemplate="Hostile Squad: Outside Zone (Gatekeep Target)<extra></extra>"
            ))
        if enemies_inside_next:
            for ey, ex in enemies_inside_next:
                cx_ring = ex * scale_x + camper_r_px * np.cos(theta)
                cy_ring = ey * scale_y + camper_r_px * np.sin(theta)
                fig.add_trace(go.Scatter(
                    x=cx_ring, y=cy_ring, mode="lines",
                    line=dict(color="rgba(249, 115, 22, 0.50)", width=1.4, dash="dash"),
                    hoverinfo="skip", showlegend=False
                ))
            ex_pts_in = [e[1] * scale_x for e in enemies_inside_next]
            ey_pts_in = [e[0] * scale_y for e in enemies_inside_next]
            fig.add_trace(go.Scatter(
                x=ex_pts_in, y=ey_pts_in,
                mode="markers",
                marker=dict(size=12, color="#f97316", symbol="diamond", line=dict(color="#ffffff", width=1.5)),
                name=f"Inside Yellow ({len(enemies_inside_next)})",
                hovertemplate="Hostile Squad: Inside Zone (Not Gatekept)<extra></extra>"
            ))

    # 5. Overlays for Step 4 (Simulate Mode)
    if current_step == 4:
        if active_strategy == "GATEKEEP ZONE PERIMETER":
            anchor_x = target_pos[1] * scale_x
            anchor_y = target_pos[0] * scale_y

            if target_pos != pos and threat_rt and threat_rt.get("found", False) and len(threat_rt.get("path", [])) > 1:
                tp = np.array(threat_rt["path"])
                fig.add_trace(go.Scatter(
                    x=tp[:, 1] * scale_x, y=tp[:, 0] * scale_y,
                    mode="lines",
                    line=dict(color="#10b981", width=3.5),
                    name="Flank to Perimeter Edge",
                    hoverinfo="skip"
                ))

            b_start = gatekeep_info.get("b_start", 240.0)
            b_end = gatekeep_info.get("b_end", 320.0)
            mean_b = gatekeep_info.get("mean_bearing", 280.0)
            primary_sec = gatekeep_info.get("primary_sector", "West")

            r_cone_px = 145.0
            xs_cone, ys_cone = generate_cone_pts(anchor_x, anchor_y, r_cone_px, b_start, b_end)

            # Translucent Amber Field of Fire Wedge
            fig.add_trace(go.Scatter(
                x=xs_cone, y=ys_cone,
                fill="toself",
                fillcolor="rgba(245, 158, 11, 0.22)",
                line=dict(color="#f59e0b", width=2.5),
                mode="lines",
                name=f"Gatekeep Cone ({int(b_start)}°–{int(b_end)}°)",
                hoverinfo="skip"
            ))

            # Aim Azimuth Line
            mid_rad = math.radians(mean_b)
            mid_x = anchor_x + r_cone_px * 1.15 * math.sin(mid_rad)
            mid_y = anchor_y - r_cone_px * 1.15 * math.cos(mid_rad)
            fig.add_trace(go.Scatter(
                x=[anchor_x, mid_x], y=[anchor_y, mid_y],
                mode="lines+text",
                line=dict(color="#fbbf24", width=2.5, dash="dash"),
                text=["", f"GATEKEEP AIM: {int(mean_b)}° ({primary_sec.upper()})"],
                textposition="top center",
                textfont=dict(color="#fef08a", size=10, family="sans-serif"),
                name="Aim Azimuth",
                hoverinfo="skip"
            ))

            # Threat Interception Vectors
            for es in gatekeep_info.get("enemy_sides", []):
                ey_px = es["grid_pos"][0] * scale_y
                ex_px = es["grid_pos"][1] * scale_x
                fig.add_trace(go.Scatter(
                    x=[anchor_x, ex_px], y=[anchor_y, ey_px],
                    mode="lines+text",
                    line=dict(color="rgba(239, 68, 68, 0.45)", width=1.5, dash="dot"),
                    text=["", f"Enemy: {es['sector']} ({es['distance_m']}m)"],
                    textposition="middle right",
                    textfont=dict(color="#fca5a5", size=9),
                    name="Threat Line",
                    hoverinfo="skip",
                    showlegend=False
                ))
        else:
            # Baseline direct path
            if base_rt and base_rt.get("found", False) and len(base_rt.get("path", [])) > 1:
                bp = np.array(base_rt["path"])
                fig.add_trace(go.Scatter(
                    x=bp[:, 1] * scale_x, y=bp[:, 0] * scale_y,
                    mode="lines",
                    line=dict(color="#ef4444", width=2.0, dash="dot"),
                    name="Direct Path (Unsafe)",
                    hoverinfo="skip"
                ))

            # Threat-aware tactical flank path
            if threat_rt and threat_rt.get("found", False) and len(threat_rt.get("path", [])) > 1:
                tp = np.array(threat_rt["path"])
                fig.add_trace(go.Scatter(
                    x=tp[:, 1] * scale_x, y=tp[:, 0] * scale_y,
                    mode="lines",
                    line=dict(color="#10b981", width=3.5),
                    name="Tactical Flank Route",
                    hoverinfo="skip"
                ))

            # Waypoints
            if waypoints and len(waypoints) > 2:
                w_names = [w[0] for w in waypoints[1:-1]]
                w_xs = [w[1][0] for w in waypoints[1:-1]]
                w_ys = [w[1][1] for w in waypoints[1:-1]]
                if w_names:
                    fig.add_trace(go.Scatter(
                        x=w_xs, y=w_ys, mode="markers+text",
                        marker=dict(size=8, color="#10b981", symbol="square", line=dict(color="#ffffff", width=1.2)),
                        text=w_names, textposition="bottom center",
                        textfont=dict(color="#6ee7b7", size=9, family="monospace"),
                        name="Waypoints"
                    ))

            # Target Destination Marker
            tgt_x = target_pos[1] * scale_x
            tgt_y = target_pos[0] * scale_y
            tgt_color, tgt_sym, tgt_label = ("#059669", "hexagon", "COMPOUND") if active_strategy == "FORTIFIED COMPOUND" else ("#ea580c", "triangle-up", "RIDGE BUNKER")

            fig.add_trace(go.Scatter(
                x=[tgt_x], y=[tgt_y],
                mode="markers+text",
                marker=dict(size=15, color=tgt_color, symbol=tgt_sym, line=dict(color="#ffffff", width=2.0)),
                text=[tgt_label], textposition="top center",
                textfont=dict(color="#f8fafc", size=10, family="sans-serif"),
                name="Target Location"
            ))

        # Animated Squad Vehicle Position along route
        if path_pts and len(path_pts) > 1 and st.session_state.get("anim_step", 0) > 0:
            step_idx = min(st.session_state["anim_step"], len(path_pts) - 1)
            anim_y, anim_x = path_pts[step_idx]
            fig.add_trace(go.Scatter(
                x=[anim_x * scale_x], y=[anim_y * scale_y],
                mode="markers+text",
                marker=dict(size=16, color="#f59e0b", symbol="circle", line=dict(color="#ffffff", width=2.5)),
                text=[f"SQUAD (T+{step_idx*2}s)"], textposition="top center",
                textfont=dict(color="#fef08a", size=11, family="Rajdhani"),
                name="Moving Squad"
            ))

    # Major Landmark Reference Labels
    lx = [norm[0] * IMG_W for norm in LANDMARKS_NORM.values()]
    ly = [norm[1] * IMG_H for norm in LANDMARKS_NORM.values()]
    lnames = list(LANDMARKS_NORM.keys())
    fig.add_trace(go.Scatter(
        x=lx, y=ly, mode="text",
        text=lnames, textposition="bottom center",
        textfont=dict(color="#64748b", size=9, family="sans-serif"),
        name="Landmarks", hoverinfo="skip"
    ))

    fig.update_layout(
        template="plotly_dark",
        height=780,
        margin=dict(l=0, r=0, b=0, t=15),
        paper_bgcolor="#08080c",
        plot_bgcolor="#08080c",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#cbd5e1", size=11),
            bgcolor="rgba(14, 14, 23, 0.88)"
        )
    )

    chart_event = st.plotly_chart(fig, key="erangel_canvas", on_select="rerun", selection_mode=["points"])

    # Canvas Click Handler
    if chart_event and "selection" in chart_event:
        sel_pts = chart_event["selection"].get("points", [])
        if sel_pts:
            raw_pt = sel_pts[0]
            clk_x = float(raw_pt.get("x", 0)) / IMG_W
            clk_y = float(raw_pt.get("y", 0)) / IMG_H
            click_tuple = (round(clk_x, 3), round(clk_y, 3))

            if click_tuple != st.session_state.get("last_click"):
                st.session_state["last_click"] = click_tuple

                if current_step == 1:
                    if st.session_state.get("zone_edit_target", "blue") == "blue":
                        st.session_state["zone_center"] = (clk_x, clk_y)
                    else:
                        st.session_state["next_zone_center"] = (clk_x, clk_y)
                        st.session_state["next_zone_manual"] = True
                    st.rerun()
                elif current_step == 2:
                    st.session_state["squad_pos"] = (clk_x, clk_y)
                    st.rerun()
                elif current_step == 3:
                    st.session_state["campers"].append((clk_x, clk_y))
                    st.rerun()
                elif current_step == 4:
                    st.session_state["squad_pos"] = (clk_x, clk_y)
                    st.rerun()

# Right Column: Operations Panel & Telemetry
with col_ops:
    if current_step == 1:
        with st.container(border=True):
            st.caption("RADAR OPERATIONS • STEP 1 OF 4")
            st.subheader("⭕ Dual Zone Configuration")
            st.markdown(
                "<span style='color: #94a3b8; font-size: 0.88rem;'>"
                "Select an active zone tab below, then click anywhere on the radar map to reposition its center."
                "</span>",
                unsafe_allow_html=True
            )

            curr_tgt = st.session_state.get("zone_edit_target", "blue")
            is_blue = (curr_tgt == "blue")
            is_yellow = (curr_tgt == "yellow")

            tab_col1, tab_col2 = st.columns(2)
            with tab_col1:
                blue_border = "border: 1.5px solid #38bdf8; box-shadow: 0 0 10px rgba(56, 189, 248, 0.35);" if is_blue else "border: 1px solid #1e293b;"
                blue_bg = "background: rgba(14, 116, 144, 0.2);" if is_blue else "background: rgba(15, 23, 42, 0.5);"
                st.markdown(
                    f"<div style='{blue_bg} {blue_border} border-radius: 14px; padding: 0.85rem 1.1rem; text-align: center; margin-bottom: 0.5rem;'>"
                    f"<span style='font-family: Plus Jakarta Sans, sans-serif; font-weight: 700; font-size: 1.05rem; color: #38bdf8; letter-spacing: 0.05em;'>🔵 BLUE PLAYZONE (Z_t)</span><br>"
                    f"<span style='font-size: 0.82rem; color: {'#e0f2fe' if is_blue else '#94a3b8'}; letter-spacing: 0.08em; text-transform: uppercase;'>{'● ACTIVE EDITING' if is_blue else 'Click to edit'}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if not is_blue:
                    if st.button("Edit Blue Playzone", key="btn_sel_blue", use_container_width=True):
                        st.session_state["zone_edit_target"] = "blue"
                        st.rerun()

            with tab_col2:
                yellow_border = "border: 1.5px solid #eab308; box-shadow: 0 0 10px rgba(234, 179, 8, 0.35);" if is_yellow else "border: 1px solid #1e293b;"
                yellow_bg = "background: rgba(161, 98, 7, 0.2);" if is_yellow else "background: rgba(15, 23, 42, 0.5);"
                st.markdown(
                    f"<div style='{yellow_bg} {yellow_border} border-radius: 14px; padding: 0.85rem 1.1rem; text-align: center; margin-bottom: 0.5rem;'>"
                    f"<span style='font-family: Plus Jakarta Sans, sans-serif; font-weight: 700; font-size: 1.05rem; color: #eab308; letter-spacing: 0.05em;'>🟡 NEXT SAFE ZONE (Z_t+1)</span><br>"
                    f"<span style='font-size: 0.82rem; color: {'#fef08a' if is_yellow else '#94a3b8'}; letter-spacing: 0.08em; text-transform: uppercase;'>{'● ACTIVE EDITING' if is_yellow else 'Click to edit'}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                if not is_yellow:
                    if st.button("Edit Yellow Next Zone", key="btn_sel_yellow", use_container_width=True):
                        st.session_state["zone_edit_target"] = "yellow"
                        st.rerun()

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            if is_blue:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.markdown(
                        f"<div class='hud-stat-box'>"
                        f"<div class='hud-stat-label'>🎯 Playzone Center</div>"
                        f"<div class='hud-val-blue'>({int(cur_center[0]*IMG_W)}, {int(cur_center[1]*IMG_H)})</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with t_col2:
                    st.markdown(
                        f"<div class='hud-stat-box'>"
                        f"<div class='hud-stat-label'>📐 Playzone Diameter</div>"
                        f"<div class='hud-val-blue'>{zone_diam_m}m <span class='hud-sub-dim'>({int(cur_radius*100)}%)</span></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='hud-stat-label'>Esports Phase Presets:</div>", unsafe_allow_html=True)
                p_cols = st.columns(5)
                if p_cols[0].button("P1 3.2k", key="s1_p1", use_container_width=True, type="primary" if int(cur_radius*100) == 40 else "secondary"):
                    st.session_state["zone_radius_pct"] = 0.40; st.rerun()
                if p_cols[1].button("P2 2.4k", key="s1_p2", use_container_width=True, type="primary" if int(cur_radius*100) == 30 else "secondary"):
                    st.session_state["zone_radius_pct"] = 0.30; st.rerun()
                if p_cols[2].button("P3 1.6k", key="s1_p3", use_container_width=True, type="primary" if int(cur_radius*100) == 20 else "secondary"):
                    st.session_state["zone_radius_pct"] = 0.20; st.rerun()
                if p_cols[3].button("P4 900m", key="s1_p4", use_container_width=True, type="primary" if int(cur_radius*100) == 11 else "secondary"):
                    st.session_state["zone_radius_pct"] = 0.11; st.rerun()
                if p_cols[4].button("P5 400m", key="s1_p5", use_container_width=True, type="primary" if int(cur_radius*100) == 5 else "secondary"):
                    st.session_state["zone_radius_pct"] = 0.05; st.rerun()

                r_slider_step = st.slider(
                    "Fine-Tune Blue Zone Radius (% Map)",
                    min_value=5, max_value=48,
                    value=int(st.session_state["zone_radius_pct"] * 100),
                    step=1,
                    key="s1_blue_r_slider"
                )
                if r_slider_step / 100.0 != st.session_state["zone_radius_pct"]:
                    st.session_state["zone_radius_pct"] = r_slider_step / 100.0
                    st.rerun()

                st.markdown("<div class='hud-stat-label' style='margin-top: 0.5rem;'>Quick Landmark Center:</div>", unsafe_allow_html=True)
                lm_col1, lm_col2 = st.columns([2.6, 1.4])
                with lm_col1:
                    sel_lm_b = st.selectbox("Landmark", list(LANDMARKS_NORM.keys()), index=2, key="s1_lm_blue", label_visibility="collapsed")
                with lm_col2:
                    if st.button("Center 🎯", key="s1_btn_lm_b", use_container_width=True):
                        st.session_state["zone_center"] = LANDMARKS_NORM[sel_lm_b]
                        st.rerun()
            else:
                t_col1, t_col2 = st.columns(2)
                with t_col1:
                    st.markdown(
                        f"<div class='hud-stat-box'>"
                        f"<div class='hud-stat-label'>🎯 Next Safe Center</div>"
                        f"<div class='hud-val-yellow'>({int(cur_next_center[0]*IMG_W)}, {int(cur_next_center[1]*IMG_H)})</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with t_col2:
                    st.markdown(
                        f"<div class='hud-stat-box'>"
                        f"<div class='hud-stat-label'>📐 Safe Zone Diameter</div>"
                        f"<div class='hud-val-yellow'>{next_zone_diam_m}m <span class='hud-sub-dim'>({int(cur_next_radius*100)}%)</span></div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                is_manual = st.session_state.get("next_zone_manual", False)
                badge_html = "<span class='hud-badge-yellow'>🖐️ Custom Manual Placement</span>" if is_manual else "<span class='hud-badge-blue'>🤖 AI Bayesian Forecaster</span>"
                b_c1, b_c2 = st.columns([2, 1.8])
                with b_c1:
                    st.markdown(f"<div style='margin-top: 0.35rem;'>{badge_html}</div>", unsafe_allow_html=True)
                with b_c2:
                    if is_manual:
                        if st.button("🤖 Reset to AI", key="s1_ai_reset", use_container_width=True):
                            st.session_state["next_zone_manual"] = False
                            st.rerun()

                st.markdown("<div class='hud-stat-label' style='margin-top: 0.5rem;'>Shrink Ratio Presets (% of Blue):</div>", unsafe_allow_html=True)
                y_p_cols = st.columns(3)
                target_60 = round(st.session_state["zone_radius_pct"] * 0.60, 2)
                target_50 = round(st.session_state["zone_radius_pct"] * 0.50, 2)
                target_40 = round(st.session_state["zone_radius_pct"] * 0.40, 2)
                if y_p_cols[0].button("60% Ratio", key="s1_yp60", use_container_width=True, type="primary" if round(cur_next_radius, 2) == target_60 else "secondary"):
                    st.session_state["next_zone_radius_pct"] = target_60; st.rerun()
                if y_p_cols[1].button("50% Ratio", key="s1_yp50", use_container_width=True, type="primary" if round(cur_next_radius, 2) == target_50 else "secondary"):
                    st.session_state["next_zone_radius_pct"] = target_50; st.rerun()
                if y_p_cols[2].button("40% Ratio", key="s1_yp40", use_container_width=True, type="primary" if round(cur_next_radius, 2) == target_40 else "secondary"):
                    st.session_state["next_zone_radius_pct"] = target_40; st.rerun()

                max_y_rad = max(5, int(st.session_state["zone_radius_pct"] * 100))
                y_slider_val = min(int(cur_next_radius * 100), max_y_rad)
                r_yellow_step = st.slider(
                    "Fine-Tune Yellow Zone Radius (% Map)",
                    min_value=3, max_value=max_y_rad,
                    value=y_slider_val,
                    step=1,
                    key="s1_yellow_r_slider"
                )
                if r_yellow_step / 100.0 != st.session_state["next_zone_radius_pct"]:
                    st.session_state["next_zone_radius_pct"] = r_yellow_step / 100.0
                    st.rerun()

                st.markdown("<div class='hud-stat-label' style='margin-top: 0.5rem;'>Quick Landmark Center:</div>", unsafe_allow_html=True)
                lm_col1, lm_col2 = st.columns([2.6, 1.4])
                with lm_col1:
                    sel_lm_y = st.selectbox("Landmark", list(LANDMARKS_NORM.keys()), index=3, key="s1_lm_yellow", label_visibility="collapsed")
                with lm_col2:
                    if st.button("Center 🎯", key="s1_btn_lm_y", use_container_width=True):
                        st.session_state["next_zone_center"] = LANDMARKS_NORM[sel_lm_y]
                        st.session_state["next_zone_manual"] = True
                        st.rerun()

            st.write("---")
            s1_b1, s1_b2 = st.columns([1, 1.8])
            with s1_b1:
                if st.button("🔄 Reset Setup", key="s1_reset_btn", use_container_width=True):
                    st.session_state["workflow_step"] = 1
                    st.session_state["campers"] = []
                    st.session_state["squad_pos"] = LANDMARKS_NORM["Gatka"]
                    st.session_state["zone_center"] = LANDMARKS_NORM["Pochinki"]
                    st.session_state["zone_radius_pct"] = 0.32
                    st.session_state["next_zone_center"] = (0.50, 0.45)
                    st.session_state["next_zone_radius_pct"] = 0.18
                    st.session_state["zone_edit_target"] = "blue"
                    st.session_state["next_zone_manual"] = False
                    st.session_state["time_left"] = 45
                    st.rerun()
            with s1_b2:
                if st.button("Next: Set Squad Location ➡️", type="primary", use_container_width=True):
                    st.session_state["workflow_step"] = 2
                    st.rerun()

    elif current_step == 2:
        with st.container(border=True):
            st.caption("RADAR OPERATIONS • STEP 2 OF 4")
            st.subheader("📍 Squad Deployment")
            st.markdown(
                "<span style='color: #94a3b8; font-size: 0.88rem;'>"
                "Click anywhere on the radar map to deploy your 4-man squad, or pick a designated drop spot below."
                "</span>",
                unsafe_allow_html=True
            )

            s2_c1, s2_c2 = st.columns(2)
            with s2_c1:
                st.markdown(
                    f"<div class='hud-stat-box'>"
                    f"<div class='hud-stat-label'>🎯 Squad Coordinates</div>"
                    f"<div class='hud-val-green'>({int(cur_squad[0]*IMG_W)}, {int(cur_squad[1]*IMG_H)})</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with s2_c2:
                st.markdown(
                    f"<div class='hud-stat-box'>"
                    f"<div class='hud-stat-label'>📏 Distance to Center</div>"
                    f"<div class='hud-val-blue'>{dist_to_center_m}m <span class='hud-sub-dim'>from center</span></div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            if in_next_zone:
                status_badge = "<span class='hud-badge-green'>🟢 SECURED IN NEXT SAFE ZONE (YELLOW)</span>"
                status_detail = "Prime defensive positioning. Ready to hold compound or perimeter gatekeep."
            elif in_zone:
                status_badge = "<span class='hud-badge-yellow'>🟡 INSIDE PLAYZONE (BLUE)</span>"
                status_detail = f"Squad is safe for now. Must rotate {dist_to_yellow_m}m into Yellow Safe Zone before collapse."
            else:
                status_badge = "<span class='hud-badge-red'>🔴 OUTSIDE PLAYZONE (BLUE LINE WARNING)</span>"
                status_detail = f"Immediate breach danger: {dist_to_edge_m}m to safe blue perimeter line."

            st.markdown(
                f"<div style='background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.35rem 0 0.8rem 0;'>"
                f"<div style='margin-bottom: 0.25rem;'>{status_badge}</div>"
                f"<span style='font-size: 0.82rem; color: #cbd5e1;'>{status_detail}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.markdown("<div class='hud-stat-label' style='margin-top: 0.4rem;'>Quick Drop Spot Spawner:</div>", unsafe_allow_html=True)
            sel_squad_s2 = st.selectbox("Major Drop Spot", list(LANDMARKS_NORM.keys()), index=1, key="s2_quick_squad", label_visibility="collapsed")
            if st.button("Spawn at Drop Spot 🎯", key="s2_spawn_btn", use_container_width=True):
                st.session_state["squad_pos"] = LANDMARKS_NORM[sel_squad_s2]
                st.rerun()

            st.write("---")
            s2_b1, s2_b2 = st.columns(2)
            if s2_b1.button("⬅️ Back: Set Zones", use_container_width=True):
                st.session_state["workflow_step"] = 1
                st.rerun()
            if s2_b2.button("Next: Place Enemies ➡️", type="primary", use_container_width=True):
                st.session_state["workflow_step"] = 3
                st.rerun()

    elif current_step == 3:
        with st.container(border=True):
            st.caption("RADAR OPERATIONS • STEP 3 OF 4")
            st.subheader("⚠️ Hostile Recon")
            st.markdown(
                "<span style='color: #94a3b8; font-size: 0.88rem;'>"
                "Click on the map to place spotted hostile camper squads. "
                "Starts with 0 enemies. Leave empty for a clean sector, or spot campers to test threat-avoidance routing."
                "</span>",
                unsafe_allow_html=True
            )

            n_campers = len(st.session_state['campers'])
            threat_badge = "<span class='hud-badge-green'>🟢 CLEAN SECTOR • 0 HOSTILE SQUADS DETECTED</span>" if n_campers == 0 else f"<span class='hud-badge-red'>⚠️ {n_campers} HOSTILE SQUADS DETECTED ON RADAR</span>"

            st.markdown(
                f"<div style='background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 8px; padding: 0.6rem 0.8rem; margin: 0.35rem 0 0.8rem 0;'>"
                f"{threat_badge}"
                f"</div>",
                unsafe_allow_html=True
            )

            c_ctrl1, c_ctrl2 = st.columns(2)
            if c_ctrl1.button("+ Random Squad", key="s3_rand_camper", use_container_width=True):
                rng_tmp = np.random.default_rng()
                free_pts = np.argwhere(~erangel_blocked)
                if len(free_pts) > 0:
                    rand_pt = free_pts[rng_tmp.integers(0, len(free_pts))]
                    st.session_state["campers"].append((float(rand_pt[1]) / N_GRID, float(rand_pt[0]) / N_GRID))
                    st.rerun()
            if c_ctrl2.button("🗑️ Clear Sector", key="s3_clear_camper", use_container_width=True):
                st.session_state["campers"] = []
                st.rerun()

            if st.session_state["campers"]:
                st.markdown("<div class='hud-stat-label' style='margin-top: 0.6rem;'>Spotted Squad Radar Coordinates:</div>", unsafe_allow_html=True)
                camper_chips = " ".join([
                    f"<span style='display: inline-block; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 4px; padding: 2px 7px; margin: 2px; font-family: monospace; font-size: 0.75rem; color: #fca5a5;'>#{i+1} ({int(cx*IMG_W)}, {int(cy*IMG_H)})</span>"
                    for i, (cx, cy) in enumerate(st.session_state["campers"][:8])
                ])
                if len(st.session_state["campers"]) > 8:
                    camper_chips += f"<span style='color: #94a3b8; font-size: 0.75rem;'> +{len(st.session_state['campers']) - 8} more</span>"
                st.markdown(f"<div style='margin-bottom: 0.6rem;'>{camper_chips}</div>", unsafe_allow_html=True)

            st.write("---")
            st.markdown("<div class='hud-stat-label'>Strategic Tactical Intent:</div>", unsafe_allow_html=True)
            strat_s3 = st.selectbox(
                "Tactical Objective",
                ["Auto (IGL)", "Fortified Compound", "Open Ridge Bunker", "Edge Gatekeep"],
                index=0,
                key="s3_strat_choice",
                label_visibility="collapsed"
            )
            st.session_state["strategy_mode"] = strat_s3

            st.write("")
            s3_b1, s3_b2 = st.columns(2)
            if s3_b1.button("⬅️ Back: Set Squad", use_container_width=True):
                st.session_state["workflow_step"] = 2
                st.rerun()
            if s3_b2.button("▶️ Simulate Tactical Route", type="primary", use_container_width=True):
                st.session_state["workflow_step"] = 4
                st.rerun()

    else:
        # Step 4: Tactical Operations Center
        with st.container(border=True):
            if not in_next_zone:
                b_color, b_bg = "#38bdf8", "rgba(56, 189, 248, 0.12)"
                b_border, b_tag = "border: 1px solid rgba(56, 189, 248, 0.35);", "⚡ PRIORITY 1: ROTATE INTO NEXT SAFE ZONE 🟡"
            elif active_strategy == "GATEKEEP ZONE PERIMETER":
                b_color, b_bg = "#fbbf24", "rgba(251, 191, 36, 0.12)"
                b_border, b_tag = "border: 1px solid rgba(251, 191, 36, 0.35);", "🛡️ PRIORITY 2: GATEKEEP TEAMS OUTSIDE SAFE ZONE"
            else:
                b_color, b_bg = "#34d399", "rgba(16, 185, 129, 0.12)"
                b_border, b_tag = "border: 1px solid rgba(16, 185, 129, 0.35);", "🟢 SECURED IN YELLOW ZONE • FORTIFY POSITION"

            st.markdown(
                f"<div style='{b_bg} {b_border} border-radius: 6px; padding: 0.35rem 0.65rem; margin-bottom: 0.45rem;'>"
                f"<span style='font-family: Rajdhani, sans-serif; font-weight: 700; font-size: 0.78rem; letter-spacing: 0.8px; color: {b_color};'>{b_tag}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.subheader(active_strategy)

            info_c1, info_c2 = st.columns(2)
            if active_strategy == "GATEKEEP ZONE PERIMETER" and gatekeep_info:
                b_s = int(gatekeep_info.get("b_start", 0))
                b_e = int(gatekeep_info.get("b_end", 0))
                pri_sec = gatekeep_info.get("primary_sector", "Perimeter").upper()
                info_c1.markdown(f"**Field of Fire**: `{b_s}°–{b_e}° ({pri_sec})`")
                info_c2.markdown(f"**Gatekeep Targets**: {len(enemies_outside_next)} Squads Outside")
            else:
                in_yellow = math.dist(target_pos, goal) <= r_next_grid
                sec_tag = " • Inside Next Safe Zone 🟡" if in_yellow else ""
                info_c1.markdown(f"**Target Grid**: `({target_pos[1]}, {target_pos[0]})`{sec_tag}")
                info_c2.markdown(f"**Terrain**: {target_terrain} ({int(erangel_cover[target_pos]*100)}% cover)")

            st.write(tactical_advice)

            if active_strategy == "GATEKEEP ZONE PERIMETER" and gatekeep_info.get("distinct_sectors"):
                st.markdown("---")
                st.markdown(f"**🛡️ Gatekeep Outside Teams ({len(enemies_outside_next)} trapped outside):**")
                for s in gatekeep_info["distinct_sectors"][:3]:
                    st.markdown(
                        f"<div style='background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; padding: 0.4rem 0.65rem; border-radius: 0 6px 6px 0; margin-bottom: 0.35rem;'>"
                        f"<span style='font-weight: 700; color: #fbbf24;'>🎯 {s['sector']} Side ({int(s['bearing'])}° • {s['distance_m']}m away)</span><br>"
                        f"<span style='font-size: 0.8rem; color: #cbd5e1;'>Trapped outside safe zone. Pre-aim DMR sightlines.</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        # Field Execution Protocols
        with st.container(border=True):
            st.caption("OPERATIONAL PLAYBOOK")
            st.subheader("Field Execution Protocols")
            for i, p in enumerate(protocols):
                st.markdown(
                    f"<div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem;'>"
                    f"<span style='font-family: Rajdhani, sans-serif; font-weight: 700; font-size: 0.85rem; color: #38bdf8;'>[0{i+1}] {p['title']}</span><br>"
                    f"<span style='color: #94a3b8; font-size: 0.84rem;'>{p['desc']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Hazard & Risk Analysis
        with st.container(border=True):
            st.caption("THREAT & RISK ANALYSIS")
            st.subheader("Vulnerabilities & Mitigations")
            for r in risks:
                st.markdown(
                    f"<div style='background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 6px; padding: 0.5rem 0.75rem; margin-bottom: 0.4rem;'>"
                    f"<span style='color: #f87171; font-weight: 700; font-size: 0.82rem;'>⚠️ HAZARD:</span> <span style='color: #f1f5f9; font-size: 0.84rem;'>{r['hazard']}</span><br>"
                    f"<span style='color: #34d399; font-weight: 700; font-size: 0.82rem;'>🛡️ MITIGATION:</span> <span style='color: #94a3b8; font-size: 0.84rem;'>{r['mitigation']}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # Movement Telemetry & Clock
        with st.container(border=True):
            st.caption("MOVEMENT TELEMETRY & CLOCK")
            st.subheader("Transit Metrics")
            s4_time = st.slider(
                "Blue Zone Collapse Timer (Seconds Remaining)",
                min_value=10, max_value=120,
                value=int(st.session_state.get("time_left", 45)),
                step=5,
                key="step4_time_slider"
            )
            if s4_time != st.session_state.get("time_left", 45):
                st.session_state["time_left"] = s4_time
                st.rerun()

            t_col1, t_col2 = st.columns(2)
            t_col1.metric("Distance", f"{distance_m}m", f"{num_steps} cells")
            t_col2.metric("Vehicle ETA", f"{vehicle_eta_sec}s", "80 km/h")

            t_col3, t_col4 = st.columns(2)
            t_col3.metric("Foot Sprint ETA", f"{foot_eta_sec}s", "22 km/h")
            t_col4.metric(
                "Safe Zone Margin",
                f"{time_left - vehicle_eta_sec:+d}s",
                "Sufficient" if time_left >= vehicle_eta_sec else "Deficit",
                delta_color="normal" if time_left >= vehicle_eta_sec else "inverse"
            )

        # Search Optimization Comparison
        with st.container(border=True):
            st.subheader("Search Optimization Comparison")
            comp_df = pd.DataFrame({
                "Metric": ["Survival Probability", "Threat Exposure", "Camper Standoff Buffer", "Expanded Search Nodes"],
                "Threat-Aware A* (Flank)": [
                    f"{threat_surv*100:.1f}%",
                    f"{threat_exp:.2f}",
                    f"{min_camper_dist_m}m" if enemies else "No Threat",
                    f"{threat_rt.get('expanded', 0)}"
                ],
                "Baseline Direct (Straight)": [
                    f"{base_surv*100:.1f}%",
                    f"{base_exp:.2f}",
                    "Intercepted" if (enemies and min_camper_dist_m < 250) else "Clear",
                    f"{base_rt.get('expanded', 0)}"
                ]
            })
            st.dataframe(comp_df, hide_index=True, width="stretch")

        # Reset / Adjustment Controls
        sim_b1, sim_b2, sim_b3 = st.columns([1, 1, 1])
        if sim_b1.button("🔄 Adjust Zones", use_container_width=True):
            st.session_state["workflow_step"] = 1
            st.rerun()
        if sim_b2.button("⚠️ Edit Campers", use_container_width=True):
            st.session_state["workflow_step"] = 3
            st.rerun()
        if sim_b3.button("🗑️ Reset Setup", use_container_width=True):
            st.session_state["workflow_step"] = 1
            st.session_state["campers"] = []
            st.session_state["squad_pos"] = LANDMARKS_NORM["Gatka"]
            st.session_state["zone_center"] = LANDMARKS_NORM["Pochinki"]
            st.session_state["zone_radius_pct"] = 0.32
            st.session_state["next_zone_center"] = (0.50, 0.45)
            st.session_state["next_zone_radius_pct"] = 0.18
            st.session_state["zone_edit_target"] = "blue"
            st.session_state["next_zone_manual"] = False
            st.session_state["time_left"] = 45
            st.rerun()


# Empirical Benchmarks & Scalability Suite Expander (Single Page)
with st.expander("📊 Empirical Benchmarks & Scalability Suite (100 Simulations)", expanded=False):
    st.markdown("""<div class="nl-section-header" style="margin-top: 0.5rem; margin-bottom: 1.2rem;">
<div class="nl-section-tag">📊 Empirical Evaluation</div>
<h3 class="nl-section-title" style="font-size: 1.5rem;">Comparative Performance & Scalability</h3>
<p style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.4rem; max-width: 820px;">
Rigorous performance benchmarks comparing Threat-Aware A* (Z.E.U.S.) against Uninformed BFS and Baseline Straight Pathing across 100 simulation episodes.
</p>
</div>""", unsafe_allow_html=True)

    bm_m1, bm_m2, bm_m3, bm_m4 = st.columns(4)
    bm_m1.metric("Survival Rate Gain", "+300%", "Threat-A* vs BFS/Manhattan")
    bm_m2.metric("Planning Latency", "< 15 ms", "Sub-millisecond on 50x50")
    bm_m3.metric("Admissibility Verified", "0 Violations", "h(n) <= h*(n) across 8,758 nodes")
    bm_m4.metric("Manhattan Flaw", "4,825 Violations", "Inadmissible on 8-connected grid")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    tab_sum, tab_topo, tab_acc = st.tabs(["📋 Summary by Scale", "🗺️ Topology Breakdown", "🎯 Forecaster Accuracy"])
    with tab_sum:
        summary_csv = "results/benchmark_summary.csv"
        if os.path.exists(summary_csv):
            st.dataframe(pd.read_csv(summary_csv), use_container_width=True, hide_index=True)
    with tab_topo:
        topo_csv = "results/benchmark_by_topology.csv"
        if os.path.exists(topo_csv):
            st.dataframe(pd.read_csv(topo_csv), use_container_width=True, hide_index=True)
    with tab_acc:
        acc_csv = "results/forecaster_accuracy.csv"
        if os.path.exists(acc_csv):
            st.dataframe(pd.read_csv(acc_csv), use_container_width=True, hide_index=True)

    st.markdown("#### 📈 Benchmark Visualizations")
    p_c1, p_c2 = st.columns(2)
    with p_c1:
        if os.path.exists("results/plot_survival.png"):
            st.image("results/plot_survival.png", caption="Survival Rate Comparison Across Topologies", use_container_width=True)
    with p_c2:
        if os.path.exists("results/plot_runtime_nodes.png"):
            st.image("results/plot_runtime_nodes.png", caption="Search Runtime & Node Expansion Scalability", use_container_width=True)
