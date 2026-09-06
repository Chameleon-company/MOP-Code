import os
import plotly.io as pio
from ipywidgets import Button, HTML
from IPython.display import display

def add_export_button(fig, file_name_prefix, export_folder="exported_plots"):
    """
    Adds an export button to the plotly figure to save the plot as an HTML file.
    
    Parameters:
    - fig: Plotly figure object.
    - file_name_prefix: Prefix for the exported file name.
    - export_folder: Folder where the files will be saved. Default is 'exported_plots'.
    """
    export_button = Button(description="Export", button_style="success", icon="download")
    output_message = HTML()

    # Ensure the export folder exists
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    def on_export_clicked(b):
        # Construct the full path for the exported file
        file_name = os.path.join(export_folder, f"{file_name_prefix}.html")
        pio.write_html(fig, file=file_name)
        output_message.value = f"Plot successfully exported as <b>{file_name}</b>."

    export_button.on_click(on_export_clicked)
    display(export_button, output_message)