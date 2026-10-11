"""
ScareX Standalone Raspberry Pi 5 - Local PDF Report Generator
Generates PDF crop intelligence and surveillance reports directly on the Pi.
Stores reports persistently in reports/ directory without requiring laptop transfer.
Supports daily reports and custom date-range queries.
"""
import os
import glob
from datetime import datetime
from fpdf import FPDF

try:
    import raspberry_pi.config as config
    from raspberry_pi.database.db import db
    from raspberry_pi.fusion.decision_engine import decision_engine
except ImportError:
    try:
        import laptop.config as config
        from laptop.database.db import db
        from laptop.fusion.decision_engine import decision_engine
    except ImportError:
        import config

class PDFReportGenerator:
    def __init__(self, reports_dir=None):
        if reports_dir:
            self.reports_dir = reports_dir
        elif hasattr(config, "REPORTS_DIR"):
            self.reports_dir = config.REPORTS_DIR
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.reports_dir = os.path.join(base_dir, "reports")

        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self, start_date=None, end_date=None, custom_title=None) -> str:
        """
        Builds a comprehensive PDF report using local SQLite logs and live telemetry.
        Returns the absolute path to the generated PDF.
        """
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        date_label = f"{start_date} to {end_date}" if (start_date and end_date) else (start_date or now.strftime("%Y-%m-%d"))

        events = db.get_events_by_date_range(start_date=start_date, end_date=end_date)
        status = decision_engine.get_status()
        tomato = status["tomato"]
        bird = status["bird"]

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Header Banner
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(24, 43, 73)
        pdf.cell(0, 10, txt="ScareX Standalone Autonomous System", ln=True, align='C')
        pdf.set_font("Arial", 'I', 11)
        pdf.set_text_color(80, 90, 110)
        pdf.cell(0, 6, txt="Raspberry Pi 5 Edge Surveillance & Crop Intelligence Report", ln=True, align='C')
        pdf.ln(4)

        # Divider line
        pdf.set_draw_color(180, 200, 220)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Metadata Box
        pdf.set_font("Arial", size=9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(95, 5, txt=f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}", ln=False)
        pdf.cell(95, 5, txt=f"Reporting Period: {date_label}", ln=True, align='R')
        pdf.cell(95, 5, txt=f"Host Platform: Raspberry Pi 5 (Edge Standalone)", ln=False)
        pdf.cell(95, 5, txt=f"Storage: Persistent SQLite Database", ln=True, align='R')
        pdf.ln(6)

        # Section 1: Tomato Crop Maturity Analytics
        pdf.set_font("Arial", 'B', 13)
        pdf.set_text_color(16, 120, 80)
        pdf.cell(0, 7, txt="1. Tomato Crop Ripeness Distribution", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(40, 40, 40)

        max_counts = tomato["session_max_counts"]
        total = tomato["total_observed"]
        pdf.cell(0, 6, txt=f"Total Observed Tomatoes: {total}", ln=True)
        pdf.cell(60, 6, txt=f"- Fully-Ripened: {max_counts['fully_ripened']} ({tomato['full_pct']}%)", ln=False)
        pdf.cell(60, 6, txt=f"- Half-Ripened:  {max_counts['half_ripened']} ({tomato['half_pct']}%)", ln=False)
        pdf.cell(60, 6, txt=f"- Green:         {max_counts['green']} ({tomato['green_pct']}%)", ln=True)
        
        pdf.set_font("Arial", 'B', 11)
        priority = tomato.get("harvest_priority", "STANDBY")
        pdf.cell(0, 7, txt=f"Recommended Harvest Priority: {priority}", ln=True)
        pdf.ln(3)

        # Row-wise Spatial Distribution Table
        if tomato.get("row_analysis"):
            pdf.set_font("Arial", 'B', 10)
            pdf.set_fill_color(230, 240, 235)
            pdf.cell(20, 6, "Row", 1, 0, 'C', True)
            pdf.cell(35, 6, "Total Observed", 1, 0, 'C', True)
            pdf.cell(35, 6, "Green (%)", 1, 0, 'C', True)
            pdf.cell(35, 6, "Half-Ripe (%)", 1, 0, 'C', True)
            pdf.cell(35, 6, "Fully-Ripe (%)", 1, 0, 'C', True)
            pdf.cell(30, 6, "Priority", 1, 1, 'C', True)

            pdf.set_font("Arial", size=9)
            for r in tomato["row_analysis"]:
                pdf.cell(20, 5, f"Row {r['row_number']}", 1, 0, 'C')
                pdf.cell(35, 5, str(r['total']), 1, 0, 'C')
                pdf.cell(35, 5, f"{r['green_pct']}%", 1, 0, 'C')
                pdf.cell(35, 5, f"{r['half_pct']}%", 1, 0, 'C')
                pdf.cell(35, 5, f"{r['full_pct']}%", 1, 0, 'C')
                pdf.cell(30, 5, str(r['priority']), 1, 1, 'C')
            pdf.ln(4)

        # Section 2: Bird Surveillance & Protection Audit
        pdf.set_font("Arial", 'B', 13)
        pdf.set_text_color(180, 40, 40)
        pdf.cell(0, 7, txt="2. Autonomous Bird Protection Audit", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.set_text_color(40, 40, 40)

        # Count deterrence events
        scare_events = [e for e in events if "DETERRENCE" in e.get("event_type", "")]
        species_counts = {}
        for e in scare_events:
            s = e.get("species", "Bird")
            species_counts[s] = species_counts.get(s, 0) + 1

        pdf.cell(0, 6, txt=f"Total Deterrence Activations: {len(scare_events)}", ln=True)
        if species_counts:
            spec_str = ", ".join([f"{k}: {v}" for k, v in species_counts.items()])
            pdf.cell(0, 6, txt=f"Targeted Species Encountered: {spec_str}", ln=True)
        pdf.ln(3)

        # Section 3: Event Log Excerpt
        pdf.set_font("Arial", 'B', 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 6, txt="Recent Security, Actuation & Crop Events:", ln=True)

        pdf.set_font("Arial", 'B', 8)
        pdf.set_fill_color(240, 240, 245)
        pdf.cell(35, 5, "Timestamp", 1, 0, 'L', True)
        pdf.cell(45, 5, "Event Type", 1, 0, 'L', True)
        pdf.cell(30, 5, "Species", 1, 0, 'L', True)
        pdf.cell(20, 5, "Confidence", 1, 0, 'C', True)
        pdf.cell(60, 5, "Robot Action", 1, 1, 'L', True)

        pdf.set_font("Arial", size=8)
        for ev in events[:18]:
            ts = str(ev.get("timestamp", ""))[:19]
            etype = str(ev.get("event_type", ""))[:25]
            spec = str(ev.get("species", "None"))[:16]
            conf = f"{float(ev.get('bird_confidence', 0.0))*100:.0f}%" if ev.get("bird_confidence") else "-"
            act = str(ev.get("robot_action", ""))[:35]

            pdf.cell(35, 4.5, ts, 1, 0, 'L')
            pdf.cell(45, 4.5, etype, 1, 0, 'L')
            pdf.cell(30, 4.5, spec, 1, 0, 'L')
            pdf.cell(20, 4.5, conf, 1, 0, 'C')
            pdf.cell(60, 4.5, act, 1, 1, 'L')

        pdf.ln(6)
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 4, txt="Report autonomously compiled by ScareX Raspberry Pi 5 Edge Intelligence Unit.", ln=True, align='C')

        filename = f"scarex_report_{timestamp_str}.pdf"
        output_path = os.path.join(self.reports_dir, filename)
        pdf.output(output_path)

        # Also maintain a symlink/copy to latest report
        latest_path = os.path.join(self.reports_dir, "scarex_latest_report.pdf")
        try:
            pdf.output(latest_path)
        except Exception:
            pass

        print(f"[PDFReportGenerator] Generated report saved to: {output_path}")
        return output_path

    def list_reports(self) -> list:
        pattern = os.path.join(self.reports_dir, "*.pdf")
        files = glob.glob(pattern)
        reports = []
        for f in sorted(files, key=os.path.getmtime, reverse=True):
            reports.append({
                "filename": os.path.basename(f),
                "path": f,
                "size_kb": round(os.path.getsize(f) / 1024, 1),
                "created_time": datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M:%S")
            })
        return reports

# Global singleton
pdf_generator = PDFReportGenerator()
