import streamlit as st
import plotly.graph_objects as go
import subprocess
import json

def call_anomaly_detection_script(data_list):
    
    list_str = ' '.join(map(str, data_list))
    
    try:
        completed_process = subprocess.run(
            ["python", "anomalydetector.py", list_str],
            text=True,  
            capture_output=True, 
            check=True,

        )
        output = completed_process.stdout.strip() 
        output = json.loads(output) 
    except subprocess.CalledProcessError as e:
        output = ''
        print("Error:", e.stderr)
    except Exception as e:
        output = ''
        print("Exeption:", e)
    return output

def create_thermometer_chart(loss, threshold):
   
    limit=100
    if limit<loss:
        limit=limit+loss
        my_color='red'
    else:
        my_color='green'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=loss,
        gauge={'axis': {'range': [None, limit]},
               'bar': {'color': my_color},
               'steps': [
                   {'range': [0, threshold], 'color': "lightgray"}],
               'threshold': {
                   'line': {'color': "green", 'width': 4},
                   'thickness': 0.75,
                   'value': threshold}}))

    fig.update_layout(
        title="Anomaly indicator",
        font=dict(
            family="Courier New, monospace",
            size=10,
            color="RebeccaPurple"
        )
    )

    return fig

# Function to call when 'Enter' is pressed
def on_enter(input_list):
    result = call_anomaly_detection_script(input_list)
    st.write(str(result))
    exit(0)
    st.write("Diagnostic:", result[0])
    thermometer_chart = create_thermometer_chart(result[1], result[2])
    st.plotly_chart(thermometer_chart)

# Application title
st.title('Anomaly detector')

# Text field for user input
with st.form(key='form'):
    input_text = st.text_area('Enter your list of values (separated by new lines):', height=300)
    submitted = st.form_submit_button(label='Submit')

# Processing user input upon form submission
if submitted:
    try:
        # Convert the text input into a list of numbers
        input_list = [float(num.replace(",", ".")) for num in input_text.split('\n') if num]
        if len(input_list) != 24:
            try:
                pass
            except:
                st.error("Error: The input must consist of 24 rows. You submitted "+len(input_list))
        # Creating the plot using Plotly
        fig = go.Figure()

        # Adding the line plot
        fig.add_trace(go.Scatter(x=list(range(len(input_list))), y=input_list,
                                 mode='lines+markers',  # It will create both lines and markers on each data point
                                 name='Values',
                                 marker=dict(color='MediumPurple')))

        # Updating layout for a clean and stylish look
        fig.update_layout(
            title='Input Values',
            xaxis_title='Time',
            yaxis_title='Value',
            template='plotly_white',  # A clean template provided by Plotly
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Call the function when 'Enter' is pressed 
        on_enter(input_list)

    except ValueError as e:
        st.write("Please ensure all entries are numbers and use the correct format.")
    except Exception as e:
        st.write("An error occurred:", e)

