"""
main.py  —  Celebrity Face Recognition System GUI

Layout (900 x 600)
------------------
  [Header]   — navy #1F4E79, title + subtitle
  [Left  ]   — uploaded photo panel
  [Right ]   — top-3 recognition results with confidence bars
  [Buttons]  — Upload Photo | View Results
  [Status]   — last action message with coloured indicator dot

Run:
    python main.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk

# ── Colour palette ────────────────────────────────────────────
C_HEADER    = '#1F4E79'   # dark navy — header background
C_BG        = '#EEF2F7'   # light blue-grey — window background
C_PANEL     = '#FFFFFF'   # white — panel backgrounds
C_DIVIDER   = '#CBD5E1'   # soft grey — dividers and empty bars
C_BTN_PRIM  = '#1F4E79'   # primary button (Upload Photo)
C_BTN_SEC   = '#2E86AB'   # secondary button (View Results)
C_TEXT      = '#1E293B'   # near-black — main text
C_SUBTEXT   = '#64748B'   # grey — section labels, secondary text
C_GREEN     = '#22C55E'   # confidence >= 70 %
C_ORANGE    = '#F97316'   # confidence 35–69 %
C_RED       = '#EF4444'   # confidence < 35 %
C_WARN      = '#F59E0B'   # low-confidence badge text
C_STATUS_BG = '#1E293B'   # status bar background

# Card background and rank label colours — three shades for visual hierarchy
C_CARD_BG   = ['#DBEAFE', '#E0F2FE', '#F0F9FF']
C_RANK_FG   = ['#1D4ED8', '#0369A1', '#0E7490']

# ── Typography ────────────────────────────────────────────────
F_TITLE  = ('Helvetica', 19, 'bold')
F_SUB    = ('Helvetica',  9)
F_SECT   = ('Helvetica', 10, 'bold')
F_NAME   = ('Helvetica', 12, 'bold')
F_NOTE   = ('Helvetica',  9, 'italic')
F_LABEL  = ('Helvetica', 10)
F_STATUS = ('Helvetica',  9)

# Pillow resampling — compatible with Pillow < 10 and >= 10
_LANCZOS = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS')

# Path constants
MODEL_PATH = os.path.join('model', 'trained_model.pkl')
CM_PATH    = os.path.join('results', 'confusion_matrix.png')


# ─────────────────────────────────────────────────────────────
class FaceRecognitionApp:
    """Main application window for the Celebrity Face Recognition System."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("900x620")
        self.root.resizable(False, False)
        self.root.configure(bg=C_BG)

        # Keep ImageTk references alive so the GC doesn't collect them
        self._photo_tk  = None
        self._cm_img_tk = None

        # Build UI sections
        self._build_header()
        self._build_content()
        self._build_buttons()
        self._build_status()

        # Clear result cards to their default empty state
        self._reset_results()

        # Warn in the status bar if the model hasn't been trained yet
        self._check_model_ready()

    # ── Layout builders ───────────────────────────────────────

    def _build_header(self) -> None:
        """Navy header bar with title and subtitle."""
        hdr = tk.Frame(self.root, bg=C_HEADER, height=68)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)

        tk.Label(hdr,
                 text="Face Recognition System",
                 bg=C_HEADER, fg='white', font=F_TITLE
                 ).place(relx=0.5, rely=0.40, anchor='center')

        tk.Label(hdr,
                 text="PCA + SVM  ·  Celebrity Dataset",
                 bg=C_HEADER, fg='#93C5FD', font=F_SUB
                 ).place(relx=0.5, rely=0.78, anchor='center')

    def _build_content(self) -> None:
        """Two-column content area: photo on the left, results on the right."""
        content = tk.Frame(self.root, bg=C_BG)
        content.pack(fill='both', expand=True, padx=12, pady=10)

        # ── Left panel: uploaded photo ─────────────────────────
        left = tk.Frame(content, bg=C_PANEL, width=380)
        left.pack(side='left', fill='y', padx=(0, 6))
        left.pack_propagate(False)

        self._panel_heading(left, "UPLOADED PHOTO")

        # Placeholder shown before any photo is loaded
        self.photo_label = tk.Label(
            left,
            text="No photo uploaded\n\nClick  \"Upload Photo\"\nto begin",
            bg='#F1F5F9', fg=C_SUBTEXT,
            font=F_LABEL, justify='center'
        )
        self.photo_label.pack(fill='both', expand=True, padx=14, pady=(0, 14))

        # ── Right panel: recognition results ───────────────────
        right = tk.Frame(content, bg=C_PANEL)
        right.pack(side='left', fill='both', expand=True)

        self._panel_heading(right, "RECOGNITION RESULTS")

        self.results_frame = tk.Frame(right, bg=C_PANEL)
        self.results_frame.pack(fill='both', expand=True, padx=14, pady=(0, 14))

        self._build_result_cards()

    def _panel_heading(self, parent: tk.Frame, text: str) -> None:
        """Section label + thin horizontal divider."""
        tk.Label(parent, text=text, bg=C_PANEL, fg=C_SUBTEXT,
                 font=F_SECT).pack(pady=(14, 0))
        tk.Frame(parent, bg=C_DIVIDER, height=1).pack(fill='x', padx=14, pady=6)

    def _build_result_cards(self) -> None:
        """
        Build three result cards (1st / 2nd / 3rd match).
        Each card holds: rank label, name label, note label (low confidence),
        percentage label, and a Canvas-based confidence bar.
        """
        rank_labels = ['1st', '2nd', '3rd']
        self.cards  = []

        for i in range(3):
            # Outer card frame with pastel background
            card = tk.Frame(self.results_frame, bg=C_CARD_BG[i])
            card.pack(fill='x', pady=(0, 10))

            inner = tk.Frame(card, bg=C_CARD_BG[i])
            inner.pack(fill='both', padx=14, pady=10)

            # ── Top row: rank + name + percentage ─────────────
            top_row = tk.Frame(inner, bg=C_CARD_BG[i])
            top_row.pack(fill='x')

            tk.Label(top_row,
                     text=rank_labels[i],
                     bg=C_CARD_BG[i], fg=C_RANK_FG[i],
                     font=('Helvetica', 10, 'bold'), width=4, anchor='w'
                     ).pack(side='left')

            name_lbl = tk.Label(top_row,
                                text="—",
                                bg=C_CARD_BG[i], fg=C_TEXT,
                                font=F_NAME, anchor='w')
            name_lbl.pack(side='left', padx=(6, 0))

            pct_lbl = tk.Label(top_row,
                               text="",
                               bg=C_CARD_BG[i], fg=C_SUBTEXT,
                               font=F_LABEL)
            pct_lbl.pack(side='right')

            # ── Note row: shown only when confidence is low ────
            note_lbl = tk.Label(inner,
                                text="",
                                bg=C_CARD_BG[i], fg=C_WARN,
                                font=F_NOTE, anchor='w')
            note_lbl.pack(fill='x', pady=(1, 0))

            # ── Confidence bar: Canvas for reliable width ──────
            bar_canvas = tk.Canvas(inner, height=10, bg=C_DIVIDER,
                                   highlightthickness=0, bd=0)
            bar_canvas.pack(fill='x', pady=(6, 0))

            self.cards.append({
                'name': name_lbl,
                'pct':  pct_lbl,
                'note': note_lbl,
                'bar':  bar_canvas,
            })

    def _build_buttons(self) -> None:
        """Upload Photo and View Results buttons."""
        frame = tk.Frame(self.root, bg=C_BG)
        frame.pack(fill='x', padx=12, pady=(0, 8))

        btn_kw = dict(
            font=('Helvetica', 11, 'bold'),
            relief='flat', bd=0,
            padx=30, pady=10,
            cursor='hand2'
        )

        tk.Button(frame,
                  text="Upload Photo",
                  bg=C_BTN_PRIM, fg='white',
                  activebackground='#163D62', activeforeground='white',
                  command=self._upload_photo,
                  **btn_kw
                  ).pack(side='left', padx=(0, 10))

        tk.Button(frame,
                  text="View Results",
                  bg=C_BTN_SEC, fg='white',
                  activebackground='#1F6D8A', activeforeground='white',
                  command=self._view_results,
                  **btn_kw
                  ).pack(side='left')

    def _build_status(self) -> None:
        """Thin status bar at the very bottom of the window."""
        bar = tk.Frame(self.root, bg=C_STATUS_BG, height=26)
        bar.pack(fill='x', side='bottom')
        bar.pack_propagate(False)

        # Coloured dot: green = ok, red = error
        self._dot = tk.Label(bar, text="●",
                             bg=C_STATUS_BG, fg='#22C55E',
                             font=('Helvetica', 9))
        self._dot.pack(side='left', padx=(10, 4))

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bar,
                 textvariable=self.status_var,
                 bg=C_STATUS_BG, fg='#94A3B8',
                 font=F_STATUS, anchor='w'
                 ).pack(side='left')

    # ── Helper utilities ──────────────────────────────────────

    def _set_status(self, msg: str, ok: bool = True) -> None:
        """Update the status bar text and indicator dot colour."""
        self.status_var.set(msg)
        self._dot.configure(fg='#22C55E' if ok else '#EF4444')
        self.root.update_idletasks()

    def _check_model_ready(self) -> None:
        """Show a warning in the status bar if the model file is missing."""
        if not os.path.exists(MODEL_PATH):
            self._set_status(
                "Model not trained yet — run train.py first", ok=False
            )
        else:
            self._set_status("Ready — upload a photo to begin")

    def _reset_results(self) -> None:
        """Clear all result cards back to their blank state."""
        for card in self.cards:
            card['name'].configure(text="—")
            card['pct'].configure(text="")
            card['note'].configure(text="")
            card['bar'].delete('all')

    def _draw_bar(self, canvas: tk.Canvas, ratio: float, colour: str) -> None:
        """
        Draw a filled confidence bar on a Canvas widget.
        ratio is a float in [0, 1]; colour is a hex colour string.
        """
        canvas.update_idletasks()
        width = canvas.winfo_width()
        if width <= 1:
            width = 460   # safe fallback before the first layout pass

        canvas.delete('all')
        canvas.create_rectangle(0, 0, int(width * ratio), 10,
                                 fill=colour, outline='')

    def _show_photo(self, path: str) -> None:
        """Load, scale and display the uploaded photo in the left panel."""
        img = Image.open(path).convert('RGB')
        img.thumbnail((350, 390), _LANCZOS)
        self._photo_tk = ImageTk.PhotoImage(img)
        self.photo_label.configure(image=self._photo_tk, text='', bg='#F1F5F9')

    def _confidence_colour(self, conf: float) -> str:
        """Return the appropriate colour for a given confidence value."""
        if conf >= 0.70:
            return C_GREEN
        elif conf >= 0.35:
            return C_ORANGE
        else:
            return C_RED

    # ── Button callbacks ──────────────────────────────────────

    def _upload_photo(self) -> None:
        """
        Open a file dialog, run recognition on the chosen photo,
        update both panels, and log the result.
        """
        # ── 1  Check model exists before opening file dialog ──
        if not os.path.exists(MODEL_PATH):
            messagebox.showinfo(
                "Model Not Found",
                "No trained model was found.\n\n"
                "Please run  train.py  first to train the model,\n"
                "then restart the application."
            )
            self._set_status("Please run train.py first.", ok=False)
            return

        # ── 2  File dialog ─────────────────────────────────────
        path = filedialog.askopenfilename(
            title="Select a photo to identify",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files",   "*.*"),
            ]
        )
        if not path:
            return   # user cancelled

        self._set_status(f"Processing  {os.path.basename(path)} ...")

        # ── 3  Display the uploaded photo ─────────────────────
        try:
            self._show_photo(path)
        except Exception as exc:
            messagebox.showerror("Image Error",
                                 f"Could not load the image:\n{exc}")
            self._set_status("Error loading image.", ok=False)
            return

        # ── 4  Run recognition ─────────────────────────────────
        try:
            from recognise import identify_person
            results = identify_person(path, top_n=3)
        except FileNotFoundError:
            messagebox.showinfo(
                "Model Not Found",
                "Trained model not found at:\n"
                f"  {MODEL_PATH}\n\n"
                "Please run  train.py  first."
            )
            self._set_status("Model missing — run train.py first.", ok=False)
            return
        except Exception as exc:
            messagebox.showerror("Recognition Error",
                                 f"Recognition failed:\n{exc}")
            self._set_status("Recognition failed — see popup.", ok=False)
            return

        # ── 5  Log result (non-fatal — never blocks the UI) ───
        try:
            from logger import log_result
            log_result(path, results)
        except Exception:
            pass

        # ── 6  Update the results panel ───────────────────────
        self._update_results(results)

        top  = results[0]
        note = "  [low confidence]" if top['low_conf'] else ""
        self._set_status(
            f"Done — best match: {top['name']}  ({top['confidence']:.1%}){note}"
        )

    def _view_results(self) -> None:
        """Open the confusion matrix image in a new child window."""
        if not os.path.exists(CM_PATH):
            messagebox.showwarning(
                "Results Not Found",
                "confusion_matrix.png was not found in results/.\n\n"
                "Please run  evaluate.py  first to generate it."
            )
            self._set_status("Run evaluate.py to generate results.", ok=False)
            return

        # Load and scale the image to fit on screen
        try:
            img = Image.open(CM_PATH)
            img.thumbnail((900, 700), _LANCZOS)
            self._cm_img_tk = ImageTk.PhotoImage(img)
        except Exception as exc:
            messagebox.showerror("Image Error",
                                 f"Could not open confusion matrix:\n{exc}")
            self._set_status("Failed to open confusion matrix.", ok=False)
            return

        # Spawn a child window
        win = tk.Toplevel(self.root)
        win.title("Confusion Matrix — Test Set")
        win.configure(bg=C_BG)
        win.resizable(True, True)

        # Mini-header
        tk.Label(win,
                 text="Confusion Matrix  —  Test Set",
                 bg=C_HEADER, fg='white', font=F_SECT
                 ).pack(fill='x', ipady=8)

        # Image
        tk.Label(win, image=self._cm_img_tk, bg=C_BG
                 ).pack(padx=10, pady=10)

        # Close button
        tk.Button(win,
                  text="Close",
                  command=win.destroy,
                  bg=C_BTN_PRIM, fg='white',
                  relief='flat',
                  font=('Helvetica', 10, 'bold'),
                  padx=20, pady=6,
                  cursor='hand2'
                  ).pack(pady=(0, 10))

        self._set_status("Confusion matrix opened.")

    def _update_results(self, results: list) -> None:
        """
        Populate the three result cards from the identify_person() output.

        Confidence colour coding:
            Green  >= 70 %
            Orange 35–69 %
            Red    < 35 %

        A '(Low confidence)' note is shown in amber on the top card when
        the best prediction probability is below the 20 % threshold.
        """
        # Let Tkinter complete any pending geometry updates so Canvas
        # widths are correct before we draw the bars
        self.root.update_idletasks()

        for i, result in enumerate(results[:3]):
            conf     = result['confidence']
            name     = result['name']
            low_conf = result.get('low_conf', False)
            card     = self.cards[i]
            colour   = self._confidence_colour(conf)

            card['name'].configure(text=name)
            card['pct'].configure(text=f"{conf:.1%}", fg=colour)

            # Show the low-confidence note only on the top card
            if low_conf and i == 0:
                card['note'].configure(text="Low confidence — treat as best guess")
            else:
                card['note'].configure(text="")

            self._draw_bar(card['bar'], conf, colour)


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    root = tk.Tk()
    FaceRecognitionApp(root)
    root.mainloop()
