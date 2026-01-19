
def get_main_css():
    return """
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Fira+Code:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f6f7fb;
            --text-color: #1f2937;
            --accent-color: #2f855a;
            --accent-light: #a7f3d0;
            --paper-shadow: 0 8px 24px rgba(31, 41, 55, 0.08);
            --border-color: #e5e7eb;
            --card-bg: #ffffff;
            --muted: #6b7280;
        }

        .stApp {
            background-color: var(--bg-color);
            font-family: 'Crimson Text', serif;
            color: var(--text-color);
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Crimson Text', serif;
            font-weight: 700;
            color: var(--text-color);
            letter-spacing: 0.2px;
        }

        .stButton button {
            background-color: var(--accent-color) !important;
            border: 1px solid var(--accent-color) !important;
            color: #ffffff !important;
            font-family: 'Fira Code', monospace !important;
            border-radius: 10px !important;
            transition: all 0.2s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.8rem;
            padding: 0.6rem 1rem !important;
        }

        .stButton button:hover {
            background-color: #276749 !important;
            border-color: #276749 !important;
            box-shadow: 0 8px 18px rgba(47, 133, 90, 0.2);
            transform: translateY(-1px);
        }

        .stSidebar {
            background-color: #ffffff;
            border-right: 1px solid var(--border-color);
        }

        .stSidebar .stSelectbox label, .stSidebar .stSlider label {
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            color: var(--muted);
        }

        .app-header {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 18px 22px;
            box-shadow: var(--paper-shadow);
            margin-bottom: 18px;
        }

        .app-title {
            font-size: 1.7rem;
            margin-bottom: 4px;
        }

        .app-subtitle {
            font-size: 0.95rem;
            color: var(--muted);
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-family: 'Fira Code', monospace;
            background: #ecfdf3;
            color: #166534;
            border: 1px solid #bbf7d0;
            margin-right: 6px;
        }

        .badge-muted {
            background: #f3f4f6;
            color: #4b5563;
            border: 1px solid #e5e7eb;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 16px;
            box-shadow: var(--paper-shadow);
        }

        .stat {
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text-color);
        }

        .stat-label {
            font-size: 0.8rem;
            color: var(--muted);
        }

        .specimen-label {
            font-family: 'Fira Code', monospace;
            font-size: 0.75rem;
            color: #6b7280;
            border-top: 1px dashed #e5e7eb;
            padding-top: 8px;
            display: flex;
            justify-content: space-between;
        }

        .specimen-id {
            color: var(--accent-color);
            font-weight: bold;
        }

    </style>
    """
