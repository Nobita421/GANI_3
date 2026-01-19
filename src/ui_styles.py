
def get_main_css():
    return """
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Fira+Code:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f1ea;
            --text-color: #2d3a3a;
            --accent-color: #4a5d23;
            --accent-light: #8fa080;
            --paper-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            --border-color: #e0ddd5;
        }

        /* Global Reset & Base */
        .stApp {
            background_color: var(--bg-color);
            background-image: url("https://www.transparenttextures.com/patterns/cream-paper.png"); /* Subtle texture if possible, else fallback color */
            font-family: 'Crimson Text', serif;
            color: var(--text-color);
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Crimson Text', serif;
            font-weight: 700;
            color: var(--text-color);
        }

        .stButton button {
            background-color: transparent !important;
            border: 2px solid var(--accent-color) !important;
            color: var(--accent-color) !important;
            font-family: 'Fira Code', monospace !important;
            border-radius: 2px !important;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-size: 0.8rem;
        }

        .stButton button:hover {
            background-color: var(--accent-color) !important;
            color: var(--bg-color) !important;
            box-shadow: 2px 2px 0px rgba(0,0,0,0.2);
            transform: translateY(-1px);
        }

        /* Sidebar */
        .stSidebar {
            background-color: #ece8e1;
            border-right: 1px solid var(--border-color);
        }
        
        .stSidebar .stSelectbox label, .stSidebar .stSlider label {
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            color: var(--accent-color);
        }

        /* Specimen Stamp Card */
        .specimen-card {
            background-color: white;
            padding: 15px;
            border: 1px solid #e6e6e6;
            box-shadow: var(--paper-shadow);
            margin-bottom: 20px;
            position: relative;
            transition: transform 0.2s;
        }
        
        .specimen-card:hover {
            transform: translateY(-2px);
            box-shadow: 4px 4px 15px rgba(0,0,0,0.1);
        }

        .specimen-img {
            width: 100%;
            border: 1px solid #eee;
            margin-bottom: 10px;
            display: block;
        }

        .specimen-label {
            font-family: 'Fira Code', monospace;
            font-size: 0.75rem;
            color: #666;
            border-top: 1px dashed #ccc;
            padding-top: 8px;
            display: flex;
            justify-content: space-between;
        }

        .specimen-id {
            color: var(--accent-color);
            font-weight: bold;
        }

        /* Tabs as Sticky Bookmarks */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: #e8e4dc;
            border-radius: 5px 5px 0 0;
            padding: 10px 20px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            border: 1px solid transparent;
            color: #888;
        }

        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: var(--bg-color);
            color: var(--accent-color);
            border: 1px solid var(--border-color);
            border-bottom: none;
            font-weight: bold;
        }
        
        /* Metric Cards */
        .metric-container {
            border: 1px solid var(--accent-color);
            padding: 1rem;
            background: #fff;
            position: relative;
        }
        
        .metric-title {
            font-family: 'Fira Code', monospace;
            font-size: 0.8rem;
            background: var(--bg-color);
            position: absolute;
            top: -0.6em;
            left: 10px;
            padding: 0 5px;
            color: var(--accent-color);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--text-color);
        }

    </style>
    """
