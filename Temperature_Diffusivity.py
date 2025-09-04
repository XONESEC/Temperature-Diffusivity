# Input Library
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
#----------------------------------------------------------------------------------------------------------------------

# Heat Diffusivity
st.header("**Temperature Diffusivity with PDE Method**")
st.subheader("Input Parameter")
input_parameter = '''In this section, the user is asked to input the values of variables that have been determined. The required variables are:

* **T0**                  = initial Temperature (K)

* **T1**                  = Temperature at Location 1 (K)

* **Tn**                  = Temperature at Location n (K)

* **k**                   = Material Conductivity (W/m.K)

* **Density**             = Material Density (Kg/m3)

* **Cp**                  = Specific Heat Capacity (J/Kg.K))

* **Length**              = Distance (m)

* **n (i)**               = the number of sections

* **dx**                  = distance between each section (m)

* **dt**                  = time  (Second)

* **nt**                  = interval per time step'''

st.markdown(input_parameter)

st.subheader("Output Parameter")
output_parameter = '''In this section, the user will obtain output in the form of graphs and tables containing the values of the predefined variables. The variable that will be displayed is:

* **Temperature** = Temperature at each section and time step'''

st.markdown(output_parameter)

st.subheader("Model")
# KASIH ILUSTRASI MODEL DALAM BENTUK GAMBAR
#----------------------------------------------------------------------------------------------------------------------

# Input Parameter
st.sidebar.title("Input")
T0 =st.sidebar.number_input("T0 (K)", value=300.000, format="%.3f")

T1 =st.sidebar.number_input("T1 (K)", value=200.000, format="%.3f")

Tn =st.sidebar.number_input("Tn (K)", value=100.000, format="%.3f")

k =st.sidebar.number_input("k (W/m.K)", value=76.100, format="%.3f")

Density =st.sidebar.number_input("Density (Kg/m3)", value=7874.000, format="%.3f")

Cp = st.sidebar.number_input("Cp (J/Kg.K)", value = 450.000, format="%.3f")

length = st.sidebar.number_input("Length (m)",min_value=0, value=100)

n = st.sidebar.number_input("n (section)",min_value=0, value=10)

dx = st.sidebar.number_input ("dx (m)", min_value = 0, value = 1 )

dt = st.sidebar.number_input("dt (second)",min_value=0, value=3600)

nt =st.sidebar.number_input ("nt (time step)", min_value= 0, value= 360)

bc_type = st.sidebar.segmented_control("Boundary Condition", ["Dirichlet", "Neumann"], selection_mode="single", default="Dirichlet")

qL, qR = 10, 10

if bc_type == "Dirichlet":
    T_left = T1
    T_right = Tn

elif bc_type == "Neumann":
    Select_Boundary = st.sidebar.segmented_control ("Boundary", ["Left (qL)", "Right (qR)"])
    if Select_Boundary == "Left (qL)":
        qL = st.sidebar.number_input("Left Boundary Gradient qL (K/m)", value=100.0, format="%.4f")
    elif Select_Boundary == "Right (qR)":
        qR = st.sidebar.number_input("Right Boundary Gradient qR (K/m)", value=100.0, format="%.4f")

dr = int(length/dx)

#----------------------------------------------------------------------------------------------------------------------

#Partial Differential Equation
st.subheader("Partial Differential Equation")
st.latex(r'''\frac{\partial T}{\partial t} = \frac{k}{\rho \, C_p} \frac{\partial^2 T}{\partial x^2}''')

st.latex(r'''\lambda = \frac{k}{\rho C_p}''')

st.latex(r'''T_{i}^{(l+1)} = \lambda \Delta t \left( \frac{T_{i+1}^{l} - 2T_{i}^{l} + T_{i-1}^{l}}{\Delta x^2} \right) + T_{i}^{l}''')
#---------------------------------------------------------------------------------------------------------------------

#Partial Diffusivity Equation

st.subheader("Dirichlet Boundary")
st.latex(r'''T_0^l = T_L, \quad T_i^l = T_R''')
st.latex(r'''T_{i}^{\,l+1} = T_{i}^{\,l} + \lambda \Delta t \left( 
\frac{T_{i+1}^{\,l} - 2T_{i}^{\,l} + T_{i-1}^{\,l}}{\Delta x^2} \right), \quad 1 \leq i \leq N-1''')

st.subheader("Neumann Boundary")
st.write ("Left Boundary")
st.latex(r'''\frac{\partial T}{\partial x}\Big|_{0} = q_L, \quad 
T_{-i}^l = T_i^l - 2 q_L \Delta x''')
st.latex(r'''T_{i}^{\,l+1} = T_{i}^{\,l} + \lambda \Delta t \left( 
\frac{T_{i+1}^{\,l} - 2T_{i}^{\,l} + (T_{i+1}^l - 2 q_L \Delta x)}{\Delta x^2} \right)''')

st.write ("Right Boundary")
st.latex(r'''\frac{\partial T}{\partial x}\Big|_{x=L} = q_R, \quad 
T_{i+1}^l = T_{i-1}^l + 2 q_R \Delta x''')
st.latex(r'''T_i^{\,l+1} = T_i^{\,l} + \lambda \Delta t \left( 
\frac{(T_{i-1}^l + 2 q_R \Delta x) - 2T_i^l + T_{i-1}^l}{\Delta x^2} \right)''')

#----------------------------------------------------------------------------------------------------------------------

# Temperature Distribution using PDE Explicit Method
def alpha (k, Density, Cp, dt):
    return ((k*dt)/(Density*Cp))
alpha_value = alpha(k, Density, Cp, dt)

temperature = np.zeros((nt+1, n+1))

def TemperatureDiffusivity(nt, n, dx, alpha_value, bc_type, qR, qL):
    for i in range(1, nt+1):
        if bc_type == "Dirichlet":
            temperature[0] = T0
            temperature[:,0] = T1
            temperature[:,-1] = Tn

            for j in range(1, n):
                temperature[i,j] = (
                    temperature[i-1,j] 
                    + (alpha_value / dx**2) * (
                        temperature[i-1,j+1] - 2*temperature[i-1,j] + temperature[i-1,j-1]
                    )
                )

        elif bc_type == "Neumann":
            temperature[0] = T0

            # Left boundary
            if Select_Boundary == "Left (qL)":
                temperature[i,0] = temperature[i,1]
            else:
                temperature[i,0] = (
                    temperature[i-1,0] 
                    + (alpha_value/ dx**2) * (
                        temperature[i-1,1] - 2*temperature[i-1,0] 
                        + (temperature[i-1,1] - 2*qL*dx)
                    )
                )

            # Right boundary
            if Select_Boundary == "Right (qR)":
                temperature[i,-1] = temperature[i,-2]
            else:
                temperature[i,-1] = (
                    temperature[i-1,-1] 
                    + (alpha_value/ dx**2) * (
                        (temperature[i-1,-2] + 2*qR*dx) - 2*temperature[i-1,-1] 
                        + temperature[i-1,-2]
                    )
                )
            for j in range(1, n):
                temperature[i,j] = (
                    temperature[i-1,j] 
                    + (alpha_value/ dx**2) * (
                        temperature[i-1,j+1] - 2*temperature[i-1,j] + temperature[i-1,j-1]
                    )
                )
    return temperature
Result = TemperatureDiffusivity(nt, n, dx, alpha_value, bc_type, qR, qL)
st.dataframe(Result)

# Visualize Temperature Distribution
st.subheader("Visualize Temperature Distribution")

x = np.arange(Result.shape[1])  
y = np.arange(Result.shape[0]) 
x, y = np.meshgrid(x, y)

fig = go.Figure(data=[go.Surface(z=Result, x=x, y=y, colorscale="Magma")])

fig.update_layout(
    scene=dict(
        xaxis_title="Number of Sections",
        yaxis_title="Time Step",
        zaxis_title="Temperature",
    ),
    autosize=True,
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

# 1D Visualization
df = pd.DataFrame(Result.T)
df.index.name = "Section"
df = df.reset_index().melt(id_vars="Section", var_name="Timestep", value_name="Temperature")

fig = px.line(
    df,
    x="Section",
    y="Temperature",
    color="Timestep",
    hover_data={"Section": True, "Temperature": True, "Timestep": True}
)

fig.update_layout(
    xaxis_title="Number of Sections",
    yaxis_title="Temperature",
    legend_title="Timestep",
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# 2D Visualization 
auto_play = st.checkbox("Auto Play", value=False)
play_speed = st.slider("Play Speed ()", 0.1, 2.0, 0.5)

plot_placeholder = st.empty()

def plot_timestep(selected_timestep):
    pressure_profile = Result[selected_timestep, :].reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 2))
    img = ax.imshow(pressure_profile, cmap='magma', aspect='auto',
                    extent=[0, Result.shape[1], 0, 1])

    cbar = fig.colorbar(img, ax=ax, shrink=1.0, aspect=5)
    cbar.set_label('Temperature (K)')

    ax.set_xlabel('Number of Sections')
    ax.set_yticks([])  
    ax.set_title(f'Temperature Distribution at Timestep {selected_timestep}')

    plot_placeholder.pyplot(fig)

if auto_play:
    for t in range(nt):
        plot_timestep(t)
        time.sleep(play_speed)
else:
    selected_timestep = st.slider("Select Timestep", min_value=0, max_value=nt-1, value=0, step=1)
    plot_timestep(selected_timestep)



