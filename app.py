import shutil
from pathlib import Path
import gradio as gr


from sbot import NOTES_DIR, build_index, graph, check_solution, drain_pending_plots

def add_pdfs_to_notes(file_paths) -> str:
    """Copy uploaded PDFs into NOTES_DIR so search_notes / read_note find them."""
    if not file_paths:
        return "_No files selected._"

    NOTES_DIR.mkdir(exist_ok=True)
    added: list[str] = []
    for f in file_paths:
        src = Path(f if isinstance(f, str) else f.name)
        if src.suffix.lower() != ".pdf":
            continue
        dest = NOTES_DIR / src.name
        shutil.copy(src, dest)
        added.append(src.name)

    if not added:
        return "_No PDFs in your upload._"

    # Re-index so semantic search picks up the new content immediately.
    build_index(force=True)
    return f"**Added {len(added)} PDF(s):** " + ", ".join(added) + " — indexed and ready."


def respond(
    message: str, chat_history: list, request: gr.Request
) -> tuple[str, list]:
    """Run a user turn through the LangGraph agent."""
    if not message.strip():
        return "", chat_history

    config = {"configurable": {"thread_id": request.session_hash}}
    result = graph.invoke({"messages": [("user", message)]}, config=config)
    reply = result["messages"][-1].content

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]
    for plot_path in drain_pending_plots():
        chat_history.append({
            "role": "assistant",
            "content": {"path": plot_path, "mime_type": "image/png"},
        })
    return "", chat_history

OLIVE_DARK = "#5c5530"
OLIVE_MID = "#6b633a"
OLIVE_LIGHT = "#827a4a"
CREAM = "#ede4cc"
CREAM_DEEP = "#f4ecd8"
INK = "#3a3520"

theme = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="amber",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Cormorant Garamond"), "Georgia", "serif"],
).set(
    body_background_fill=f"linear-gradient(160deg, {OLIVE_DARK} 0%, {OLIVE_MID} 60%, {OLIVE_LIGHT} 100%)",
    body_text_color=CREAM,
    block_background_fill="rgba(255, 250, 220, 0.06)",
    block_border_color="rgba(237, 228, 204, 0.18)",
    block_label_text_color=CREAM,
    block_title_text_color=CREAM,
    input_background_fill=CREAM_DEEP,
    input_border_color="rgba(237, 228, 204, 0.35)",
    input_placeholder_color="#7a715a",
    button_primary_background_fill=INK,
    button_primary_background_fill_hover="#54492c",
    button_primary_text_color=CREAM,
)

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500&display=swap');

.gradio-container {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    color: {CREAM} !important;
}}

/* Hero title: serif, wide letter-spacing, uppercase */
#fsf-title h1 {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: 0.32em !important;
    text-transform: uppercase !important;
    text-align: center !important;
    font-size: 2.4rem !important;
    color: {CREAM} !important;
    margin: 1.8rem 0 0.4rem 0 !important;
}}
#fsf-subtitle p {{
    text-align: center !important;
    font-style: italic !important;
    letter-spacing: 0.08em !important;
    color: {CREAM} !important;
    opacity: 0.85;
    margin-bottom: 2rem !important;
}}

/* Section labels — small caps, letter-spaced, very calm */
.fsf-section h3 {{
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.22em !important;
    text-transform: uppercase !important;
    font-size: 0.85rem !important;
    color: {CREAM} !important;
    opacity: 0.8;
    margin-bottom: 0.6rem !important;
}}

/* Inputs: cream panel with dark ink text */
.gradio-container textarea,
.gradio-container input[type="text"] {{
    color: {INK} !important;
    background: {CREAM_DEEP} !important;
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    font-size: 1.05rem !important;
}}

/* Chat panel: cream background, dark ink text inside */
.fsf-chat .chatbot,
.fsf-chat [class*="chatbot"] {{
    background: {CREAM_DEEP} !important;
    border-radius: 4px !important;
}}
.fsf-chat .bubble,
.fsf-chat [class*="bubble"],
.fsf-chat [class*="message"] {{
    color: {INK} !important;
}}
.fsf-chat .prose,
.fsf-chat .prose * {{
    color: {INK} !important;
}}

/* File uploader: subtle olive panel */
.fsf-upload [class*="upload"] {{
    background: rgba(237, 228, 204, 0.08) !important;
    border: 1px dashed rgba(237, 228, 204, 0.4) !important;
}}

/* Status text under uploader stays cream */
.fsf-upload-status p,
.fsf-upload-status em {{
    color: {CREAM} !important;
    font-style: italic !important;
    opacity: 0.9;
}}

/* Tighten footer */
footer {{ opacity: 0.5; }}
"""

with gr.Blocks(title="AI Studybot") as demo:
    gr.Markdown("# AI Studybot", elem_id="fsf-title")
    gr.Markdown("a minimalist AI helper", elem_id="fsf-subtitle")

    with gr.Row():
        with gr.Column(scale=1, elem_classes="fsf-upload"):
            gr.Markdown("### I. Notes", elem_classes="fsf-section")
            uploader = gr.File(
                label="Drop PDFs here",
                file_types=[".pdf"],
                file_count="multiple",
            )
            upload_status = gr.Markdown(
                "_Add your study PDFs to begin._",
                elem_classes="fsf-upload-status",
            )
            uploader.upload(add_pdfs_to_notes, uploader, upload_status)

            gr.Markdown("### III. Check Solution", elem_classes="fsf-section")
            solution_img = gr.Image(
                label="Photo of your handwritten work",
                type="filepath",
                height=200,
            )
            solution_btn = gr.Button("Check my work", variant="primary")

        with gr.Column(scale=3, elem_classes="fsf-chat"):
            gr.Markdown("### II. Conversation", elem_classes="fsf-section")
            chatbot = gr.Chatbot(
                height=540,
                label=None,
                show_label=False,
                # Render LaTeX so math (\lambda, fractions, etc.) shows properly.
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                    {"left": "\\[", "right": "\\]", "display": True},
                    {"left": "\\(", "right": "\\)", "display": False},
                ],
            )
            msg = gr.Textbox(
                placeholder="Ask about your notes, a math problem, or anything you're studying…",
                show_label=False,
                autofocus=True,
            )
            msg.submit(respond, [msg, chatbot], [msg, chatbot])

    # Wire the solution-check button now that `chatbot` exists.
    solution_btn.click(check_solution, [solution_img, chatbot], [chatbot, solution_img])


if __name__ == "__main__":
    print("Building notes index…")
    build_index(force=True)
    print("Index ready.")
    demo.launch(theme=theme, css=CSS, ssr_mode=False)
