import streamlit as st
import io
import torch
import json
import os
import sys
import glob
import random
import uuid
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from torchvision.transforms import ToPILImage

# Ensure src is in path
sys.path.append(os.getcwd())

# Import Backend Modules
# Try/Except to handle potential missing modules during dev
try:
    from src.inference import load_model
    from src.ui_styles import get_main_css
    from src.monitoring import monitor
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Field Notebook",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Styles
st.markdown(get_main_css(), unsafe_allow_html=True)

# Helper Paths
NOTEBOOK_DIR = "docs/notebook_entries"
FIGURES_DIR = "figures"

# Make sure notebook directory exists
os.makedirs(NOTEBOOK_DIR, exist_ok=True)


# --- Model Loading ---
@st.cache_resource
def load_engine():
    # Attempt to find the latest checkpoint
    try:
        return load_model(config_path="configs/trainconfig.yaml")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

engine = load_engine()

# --- Helpers ---
def render_header(title, subtitle, badges=None):
    badges_html = "".join(
        [f"<span class='badge {cls}'>{text}</span>" for text, cls in (badges or [])]
    )
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title">{title}</div>
            <div class="app-subtitle">{subtitle}</div>
            <div style="margin-top: 8px;">{badges_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_cards(stats):
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats):
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="stat">{value}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
def save_notebook_entry(crop, disease, count, seed, images):
    entry_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()
    
    # Save Images
    image_paths = []
    for idx, img in enumerate(images):
        filename = f"{entry_id}_{idx}.png"
        path = os.path.join(NOTEBOOK_DIR, filename)
        img.save(path)
        image_paths.append(filename)
    
    # Save Metadata
    metadata = {
        "id": entry_id,
        "timestamp": timestamp,
        "crop": crop,
        "disease": disease,
        "count": count,
        "seed": seed,
        "images": image_paths
    }
    
    json_path = os.path.join(NOTEBOOK_DIR, f"entry_{entry_id}.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    return entry_id

def mock_registry_data():
    return [
        {"version": "v1.0", "date": "2023-10-01", "fid": 45.2, "is": 2.1, "notes": "Baseline DCGAN, Tomato Only"},
        {"version": "v1.1", "date": "2023-11-15", "fid": 38.5, "is": 2.4, "notes": "Added Augmentation, Potato support"},
        {"version": "v2.0-beta", "date": "2024-01-05", "fid": 29.8, "is": 2.9, "notes": "Refined architectures, full dataset"},
    ]


# --- Page Functions ---

def render_notebook():
    render_header(
        "CropLeaf Studio",
        "Generate synthetic leaf disease specimens with curated controls and research-ready outputs.",
        badges=[
            ("GAN • DCGAN", "badge"),
            ("Dataset Augmentation", "badge badge-muted"),
        ],
    )

    last_batch = len(st.session_state.get('last_generated', []))
    status_text = "Online" if engine else "Offline (Demo)"
    render_stat_cards([
        ("Inference Engine", status_text),
        ("Last Batch", f"{last_batch} images"),
        ("Notebook Entries", len(glob.glob(os.path.join(NOTEBOOK_DIR, "entry_*.json"))))
    ])
    
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown("### Configuration")
        with st.container():
            with st.form("generate_form"):
                crop = st.selectbox("Specimen Crop", ["Tomato", "Potato", "Corn"], help="Select the plant species.")
                disease = st.selectbox("Disease Condition", ["Early Blight", "Late Blight", "Healthy", "Rust"], help="Target pathology.")

                count = st.slider("Batch Size", 1, 24, 6, help="Number of specimens to generate.")

                with st.expander("Advanced Options"):
                    seed_input = st.text_input("Seed (Optional)", placeholder="Random", help="Fixed seed for reproducibility.")
                    grid_cols = st.slider("Gallery Columns", 2, 6, 4)

                submit = st.form_submit_button("🌱 Generate Specimens", use_container_width=True)

            if submit:
                if not engine:
                    st.warning("Model not loaded. Running in Demo Mode (Noise/Random).")
                    fake_images = [Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)) for _ in range(count)]
                else:
                    try:
                        if seed_input.isdigit():
                            from src.utils import set_seed
                            set_seed(int(seed_input))

                        tensor_images = engine.generate(count)
                        to_pil = ToPILImage()
                        fake_images = [to_pil(img) for img in tensor_images]

                        monitor.log_request(crop, disease, count)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        fake_images = []

                st.session_state['last_generated'] = fake_images
                st.session_state['last_params'] = {
                    'crop': crop,
                    'disease': disease,
                    'count': count,
                    'seed': seed_input,
                    'grid_cols': grid_cols
                }

    with col2:
        st.markdown("### Specimen Board")
        
        if 'last_generated' in st.session_state:
            images = st.session_state['last_generated']
            params = st.session_state['last_params']
            
            # Grid Layout
            grid_cols = params.get('grid_cols', 4)
            cols = st.columns(grid_cols)
            for idx, img in enumerate(images):
                with cols[idx % grid_cols]:
                    # Card HTML for styling
                    # Convert img to base64 for embedding in HTML if needed, but st.image is easier.
                    # We'll use st.image inside a styled container concept slightly hacked via markdown or just use standard st elements with CSS classes.
                    
                    st.image(img, use_container_width=True) 
                    st.markdown(f"""
                    <div class="specimen-label">
                        <span>ID: <span class="specimen-id">{str(uuid.uuid4())[:6].upper()}</span></span>
                        <span>{params['crop'][:3].upper()}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # Actions
            act_col1, act_col2, act_col3 = st.columns([1, 1, 1])
            with act_col1:
                if st.button("📎 Attach to Entry", help="Save these specimens to your local notebook."):
                    entry_id = save_notebook_entry(params['crop'], params['disease'], params['count'], params['seed'], images)
                    st.success(f"Saved entry #{entry_id} to docs/notebook_entries/")
            
            with act_col2:
                 # Zip Download
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for i, img in enumerate(images):
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        zf.writestr(f"specimen_{i+1}.png", img_byte_arr.getvalue())
                
                st.download_button(
                    label="💾 Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"specimens_{params['crop']}_{datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip"
                )

            with act_col3:
                if st.button("🧹 Clear", help="Clear current batch from the board."):
                    st.session_state.pop('last_generated', None)
                    st.session_state.pop('last_params', None)

        else:
            st.info("Configure variables and click 'Cultivate Specimens' to begin.")


def render_lightbox():
    render_header(
        "Lightbox",
        "Inspect synthetic specimens with histograms and reference comparison.",
        badges=[("Inspection", "badge"), ("Nearest Neighbor", "badge badge-muted")],
    )
    
    if 'last_generated' not in st.session_state:
        st.warning("No specimens in cache. Go to 'Notebook' and generate some first.")
        return

    images = st.session_state['last_generated']
    selected_idx = st.selectbox("Select Specimen to Inspect", range(len(images)), format_func=lambda x: f"Specimen #{x+1}")
    
    col1, col2 = st.columns(2)
    
    target_img = images[selected_idx]
    
    with col1:
        st.markdown("#### Synthetic Specimen")
        st.image(target_img, use_container_width=True)
        
        # Histograms
        if st.checkbox("Show Histogram Analysis", value=False):
            # Simple histogram
            img_np = np.array(target_img)
            fig, ax = plt.subplots(figsize=(6, 2))
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                hist, bins = np.histogram(img_np[:,:,i], bins=256, range=[0,256])
                ax.plot(hist, color=color, alpha=0.7)
            ax.set_title("Color Distribution")
            ax.axis('off')
            st.pyplot(fig)

    with col2:
        st.markdown("#### Nearest Reference (Placeholder)")
        st.info("Nearest neighbor retrieval not connected to vector store.")
        # Placeholder
        st.image("https://via.placeholder.com/256?text=Reference+unavailable", use_container_width=True)


def render_metrics():
    render_header(
        "Metrics Bench",
        "Track FID/IS curves, training losses, and evaluation artifacts.",
        badges=[("FID/IS", "badge"), ("Training Logs", "badge badge-muted")],
    )
    
    # Check figures dir
    figures = glob.glob(os.path.join(FIGURES_DIR, "*.png")) + glob.glob(os.path.join(FIGURES_DIR, "*.jpg"))
    
    if not figures:
        st.info("No training artifacts found in `figures/`. Run training to populate this bench.")
        
        # Mock Charts for visual test
        st.markdown("#### Example Mock Data (Training In progress)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**FID Score (Lower is better)**")
            st.line_chart([45, 42, 38, 35, 32, 29])
        with c2:
            st.markdown("**IS Score (Higher is better)**")
            st.line_chart([1.5, 1.8, 2.1, 2.3, 2.5, 2.6])
    else:
        st.image(figures, width=300, caption=[os.path.basename(f) for f in figures])

def render_registry():
    render_header(
        "Registry & Provenance",
        "Versioned models, metadata, and usage logs for governance.",
        badges=[("Model Registry", "badge"), ("Usage Logs", "badge badge-muted")],
    )
    
    st.markdown("### Version History")
    versions = mock_registry_data()
    
    for v in versions:
        with st.container():
            c1, c2, c3 = st.columns([1, 1, 3])
            c1.markdown(f"**{v['version']}**")
            c2.markdown(f"*{v['date']}*")
            c3.markdown(f"FID: `{v['fid']}` | IS: `{v['is']}` | {v['notes']}")
            st.divider()
            
    st.markdown("### Recent Activity Log")
    log_file = "usage_logs.json"
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
            
            if logs:
                st.dataframe(logs, use_container_width=True)
            else:
                st.write("No activity logs yet.")
        except:
            st.error("Error reading log file.")
    else:
        st.write("Log file not initialized.")

# --- Main Navigation ---

def main():
    with st.sidebar:
        st.title("🍂 CropLeaf Lab")
        st.caption(f"v1.0.0 | {datetime.now().strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        page = st.radio("Navigation", ["Notebook", "Lightbox", "Metrics", "Registry"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("**System Health**")
        status_color = "green" if engine else "red"
        status_text = "Online" if engine else "Offline (Demo)"
        st.markdown(f"Inference Engine: :{status_color}[{status_text}]")
        
    if page == "Notebook":
        render_notebook()
    elif page == "Lightbox":
        render_lightbox()
    elif page == "Metrics":
        render_metrics()
    elif page == "Registry":
        render_registry()

if __name__ == "__main__":
    main()
