import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def generate_excel_report(data_dict, output_path=None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'report.xlsx'))
    """
    Generates an Excel report summarizing pipeline parameters and predicted risk.
    """
    df = pd.DataFrame([data_dict])
    df.to_excel(output_path, index=False)
    return output_path

def generate_pdf_report(data_dict, output_path=None):
    if output_path is None:
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'reports', 'report.pdf'))
    """
    Generates a PDF summary report outlining the pipeline vulnerability evaluation.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "RSPM: Pipeline Risk Analysis Report")
    
    c.setFont("Helvetica", 12)
    y_position = height - 100
    
    # Loop over fields securely and print iteratively with paging functionality logic
    for key, value in data_dict.items():
        # Clean up the key names for display
        display_key = str(key).replace('_', ' ')
        
        # Depending on value type format float
        if isinstance(value, float):
            val_str = f"{value:.4f}"
        else:
            val_str = str(value)
            
        c.drawString(50, y_position, f"{display_key}: {val_str}")
        y_position -= 25
        
        # New page if exceeding bounds
        if y_position < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 50

    c.save()
    return output_path
