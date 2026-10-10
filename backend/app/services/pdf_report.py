import os
from datetime import datetime
from typing import Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from backend.app.services.audio_analysis import recording_warning

FILLER_NOTE = "Filler counts are experimental — speech-to-text usually removes 'um/uh'. Not used in scoring."

def generate_pdf_report(evaluation: Any, concept_title: str, output_path: str) -> str:
    """
    Generates a professional PDF report containing the score, transcript,
    individual metrics, and qualitative AI feedback.
    """
    # Create the document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#1a1c29'),
        spaceAfter=12
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#3a0ca3'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2b2d42'),
        spaceAfter=8
    )

    transcript_style = ParagraphStyle(
        'TranscriptBody',
        parent=body_style,
        fontName='Helvetica-Oblique',
        textColor=colors.HexColor('#4a4e69'),
        leftIndent=15,
        rightIndent=15
    )

    card_label_style = ParagraphStyle(
        'CardLabel',
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.white,
        alignment=1 # Center
    )

    card_val_style = ParagraphStyle(
        'CardValue',
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.white,
        alignment=1 # Center
    )

    story = []
    
    # Header Banner
    story.append(Paragraph("VBCUA Assessment Report", title_style))
    story.append(Paragraph(f"Voice-Based Concept Understanding Analyser | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # Divider line
    divider = Table([[""]], colWidths=[doc.width], rowHeights=[2])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#bd00ff')),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 0.2 * inch))
    
    # Overview Table (Concept, Grade, Duration)
    overview_data = [
        [Paragraph("<b>Concept Explained:</b>", body_style), Paragraph(concept_title, body_style)],
        [Paragraph("<b>Understanding Grade:</b>", body_style), Paragraph(f"<b>{evaluation.understanding_grade}</b>", body_style)],
        [Paragraph("<b>Speech Duration:</b>", body_style), Paragraph(f"{evaluation.speech_duration:.1f} seconds", body_style)]
    ]
    overview_table = Table(overview_data, colWidths=[2.0 * inch, 4.0 * inch])
    overview_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 0.25 * inch))
    
    # Metric KPI Cards Layout
    # Three columns: Overall Score, Semantic Score, Fluency Score
    kpi_data = [
        [
            Paragraph("Overall Score", card_label_style),
            Paragraph("Semantic Similarity", card_label_style),
            Paragraph("Speech Fluency", card_label_style)
        ],
        [
            Paragraph(f"{evaluation.final_score:.1f}%", card_val_style),
            Paragraph(f"{evaluation.semantic_score:.1f}%", card_val_style),
            Paragraph(f"{100.0 - (evaluation.filler_words_ratio * 4.0):.1f}%", card_val_style) # Approximated fluency
        ]
    ]
    
    # Update fluency display logic based on evaluations
    # Wait, the evaluation object itself has the metrics saved, so let's calculate the fluency score stored
    # We will pass the exact values or retrieve them from evaluation
    # Let's adjust fluency display to show the pause ratio and filler count:
    pause_ratio = evaluation.pause_ratio_v1 if evaluation.pause_ratio_v1 is not None else evaluation.pause_ratio
    kpi_data[1][2] = Paragraph(f"P: {pause_ratio*100:.0f}% / Fillers (exp.): {evaluation.filler_words_count}", card_val_style)

    kpi_table = Table(kpi_data, colWidths=[2.1 * inch, 2.1 * inch, 2.1 * inch], rowHeights=[20, 35])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1a1c29')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#bd00ff')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2b2d42')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 0.25 * inch))
    
    # Speech Transcription Section
    story.append(Paragraph("Spoken Transcription", section_heading))
    story.append(Paragraph(f'"{evaluation.transcription}"', transcript_style))
    story.append(Spacer(1, 0.2 * inch))

    # Detailed Speech Metrics
    story.append(Paragraph("Fluency & Delivery Metrics", section_heading))
    metrics_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Target</b>", body_style)],
        [Paragraph("Pause Ratio (between first and last speech)", body_style), Paragraph(f"{pause_ratio*100:.1f}%", body_style), Paragraph("10% - 25%", body_style)],
        [Paragraph("Filler Word Count (experimental)", body_style), Paragraph(str(evaluation.filler_words_count), body_style), Paragraph("not scored", body_style)],
        [Paragraph("Filler Words Ratio (experimental)", body_style), Paragraph(f"{evaluation.filler_words_ratio:.1f} per 100 words", body_style), Paragraph("not scored", body_style)],
    ]
    metrics_table = Table(metrics_data, colWidths=[2.2 * inch, 2.1 * inch, 2.1 * inch])
    metrics_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f8fafc')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(FILLER_NOTE, ParagraphStyle('Note', parent=body_style, fontSize=8, textColor=colors.HexColor('#64748b'))))

    # Microphone check (from raw RMS; not part of any score)
    warning = recording_warning(evaluation.rms_energy)
    if warning:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Paragraph(f"<b>Note:</b> {warning}", ParagraphStyle('Warning', parent=body_style, textColor=colors.HexColor('#b45309'))))
    story.append(Spacer(1, 0.25 * inch))

    # Educational Assessment & AI Feedback
    story.append(Paragraph("AI Tutoring & Feedback", section_heading))
    
    # Convert markdown feedback from database into standard Paragraph blocks
    feedback_text = evaluation.feedback
    paragraphs = feedback_text.split('\n')
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        
        # Simple Markdown parsing
        # Headers
        if p_clean.startswith('### '):
            story.append(Paragraph(p_clean[4:], ParagraphStyle('H3', parent=styles['Heading3'], textColor=colors.HexColor('#3a0ca3'), spaceBefore=8, spaceAfter=4)))
        elif p_clean.startswith('#### '):
            story.append(Paragraph(p_clean[5:], ParagraphStyle('H4', parent=styles['Heading4'], textColor=colors.HexColor('#2b2d42'), spaceBefore=6, spaceAfter=3)))
        elif p_clean.startswith('- '):
            story.append(Paragraph(f"&bull; {p_clean[2:]}", ParagraphStyle('Bullet', parent=body_style, leftIndent=10)))
        elif p_clean.startswith('1. ') or p_clean.startswith('2. ') or p_clean.startswith('3. '):
            story.append(Paragraph(p_clean, ParagraphStyle('List', parent=body_style, leftIndent=10)))
        else:
            # Handle inline bolding (rough regex replacement)
            import re
            p_clean = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', p_clean)
            story.append(Paragraph(p_clean, body_style))
            
    # Build Document
    doc.build(story)
    return output_path
