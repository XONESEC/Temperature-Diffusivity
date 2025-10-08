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
st.header("**Heat Equation with PDE Method**", divider="gray")
st.subheader(" **Properties** ")
st.write( "In this section, the properties that will be used in the calculation are defined. The required properties are:")

kolom1 = '''
* **BC Left**       = Boundary condition value (°C for Dirichlet, °C/m for Neumann)

* **BC Right**      = Boundary condition value (°C for Dirichlet, °C/m for Neumann)

* **T0**            = Initial temperature (°C)

* **K**             = Thermal conductivity (W/(m.K))

* **cp**            = Specific heat capacity (J/(kg.K))

'''

kolom2 = '''
* **Density**       = Material density (kg/m3)

* **Alpha**         = Thermal diffusivity (m2/s)

* **Lenght**        = Domain length (m)

* **Grid number**   = Total grid

* **Time**          = Total simulation duration (s)

* **dt**            = Time step size (s)

'''

col1, col2 = st.columns(2)
with col1:
    st.markdown(kolom1)

with col2:
    st.markdown(kolom2)

#----------------------------------------------------------------------------------------------------------------------

# Input Parameter
st.sidebar.title("Boundary")

#Left Boundary
left_boundary =st.sidebar.radio("**Left Boundary :**",
                 ["Dirichlet",
                 "Neumann"])
if left_boundary == "Dirichlet":
    TL = st.sidebar.number_input("TL (K)", value=300.000, format="%.3f")

elif left_boundary == "Neumann":
    qL = st.sidebar.number_input(" qL (°C/m)", value=100.0, format="%.4f")   

#Right Boundary
right_boundary =st.sidebar.radio("**Right Boundary :**",
                 ["Dirichlet",
                 "Neumann"])
if right_boundary == "Dirichlet":
    TR = st.sidebar.number_input("TR (K)", value=300.000, format="%.3f")

elif right_boundary == "Neumann":
    qR = st.sidebar.number_input(" qR (°C/m)", value=100.0, format="%.4f") 

st.sidebar.title("Solution")
solution = st.sidebar.radio("**Solution :**",["Explicit", "Implicit"])

st.sidebar.title("Properties")

T0 =st.sidebar.number_input("T0 (°C)", value=100.000, format="%.3f")

k =st.sidebar.number_input("k (W/m.K)", value=50.000, format="%.3f")

Density =st.sidebar.number_input("Density (Kg/m3)", value=7750.000, format="%.3f")

Cp = st.sidebar.number_input("Cp (J/Kg.K)", value = 510.000, format="%.3f")

length = st.sidebar.number_input("Length (m)",min_value=0, value=10)

n = st.sidebar.number_input("Grid number",min_value=0, value=100)

total_time = st.sidebar.number_input("Total time",min_value=0, value=50000 )

dt =st.sidebar.number_input ("nt (time step)", min_value= 0, value= 400)

if dt == 0:
    r_target =0.4

alpha = int(k/(Density * Cp))


#----------------------------------------------------------------------------------------------------------------------

# Persamaan Diferensial Parsial
if solution == "Explicit":
    st.header("**1D Heat Equation (FTCS Explicit Scheme)**")
    st.subheader("Partial Differential Equation")
    st.latex(r'''
    \frac{\partial T}{\partial t} = 
    \alpha \, \frac{\partial^2 T}{\partial x^2}
    ''')

    st.latex(r'''
    \alpha = \frac{k}{\rho \, C_p}
    ''')

    st.latex(r'''
    T_i^{\,n+1} = 
    T_i^{\,n} + 
    r \left( 
    T_{i+1}^{\,n} - 2T_i^{\,n} + T_{i-1}^{\,n} 
    \right),
    \quad 
    r = \frac{\alpha \Delta t}{\Delta x^2}
    ''')

    # Kondisi Batas: Dirichlet & Neumann

    st.subheader("Dirichlet Boundary Conditions")
    st.latex(r'''
    T_0^{\,n} = T_{\text{left}}, 
    \quad 
    T_N^{\,n} = T_{\text{right}}
    ''')
    st.latex(r'''
    T_i^{\,n+1} = 
    T_i^{\,n} + 
    r \left( 
    T_{i+1}^{\,n} - 2T_i^{\,n} + T_{i-1}^{\,n} 
    \right),
    \quad 1 \le i \le N-1
    ''')

    # Neumann Boundary Conditions

    st.subheader("Neumann Boundary Conditions")

    st.write("**Left Boundary (Flux at x=0)**")
    st.latex(r'''
    \left.\frac{\partial T}{\partial x}\right|_{x=0} = q_L
    \quad \Rightarrow \quad
    T_0^{\,n+1} = T_1^{\,n+1} - q_L \, \Delta x
    ''')

    st.write("**Right Boundary (Flux at x=L)**")
    st.latex(r'''
    \left.\frac{\partial T}{\partial x}\right|_{x=L} = q_R
    \quad \Rightarrow \quad
    T_N^{\,n+1} = T_{N-1}^{\,n+1} + q_R \, \Delta x
    ''')

    # Kondisi Stabilitas

    st.subheader("Stability Condition")
    st.latex(r'''
    r = \frac{\alpha \Delta t}{\Delta x^2} \le \frac{1}{2}
    ''')

elif solution == "Implicit":
    st.header("**Fully implicit (Backward Euler) solver for 1D heat conduction**")

    # Persamaan Diferensial Parsial
    st.subheader("Partial Differential Equation")
    st.latex(r'''
    \frac{\partial T}{\partial t} = 
    \alpha \, \frac{\partial^2 T}{\partial x^2}
    ''')

    st.latex(r'''
    \alpha = \frac{k}{\rho \, C_p}
    ''')

    # Skema Implisit (Backward Euler)
    st.subheader("Fully Implicit (Backward Euler) Discretization")
    st.latex(r'''
    \frac{T_i^{\,n+1} - T_i^{\,n}}{\Delta t} =
    \alpha \, \frac{T_{i+1}^{\,n+1} - 2T_i^{\,n+1} + T_{i-1}^{\,n+1}}{(\Delta x)^2}
    ''')

    st.latex(r'''
    - r \, T_{i-1}^{\,n+1} + (1 + 2r) T_i^{\,n+1} - r \, T_{i+1}^{\,n+1} = T_i^{\,n},
    \quad
    r = \frac{\alpha \, \Delta t}{(\Delta x)^2}
    ''')

    # Sistem Matriks Linear
    st.subheader("Matrix Form")
    st.latex(r'''
    \mathbf{A} \, \mathbf{T}^{\,n+1} = \mathbf{T}^{\,n}
    ''')

    st.latex(r'''
    \mathbf{A} =
    \begin{bmatrix}
    a_{00} & a_{01} & 0      & \cdots & 0 \\
    a_{10} & a_{11} & a_{12} & \cdots & 0 \\
    0      & \ddots & \ddots & \ddots & 0 \\
    0      & \cdots & a_{N-2,N-3} & a_{N-2,N-2} & a_{N-2,N-1} \\
    0      & \cdots & 0 & a_{N-1,N-2} & a_{N-1,N-1}
    \end{bmatrix}
    ''')

    st.latex(r'''
    a_{i,i-1} = -r, \quad
    a_{i,i}   = 1 + 2r, \quad
    a_{i,i+1} = -r
    ''')

    # Kondisi Batas: Dirichlet
    st.subheader("Dirichlet Boundary Conditions")
    st.latex(r'''
    T_0^{\,n+1} = T_{\text{left}}, 
    \quad 
    T_N^{\,n+1} = T_{\text{right}}
    ''')

    # Kondisi Batas: Neumann
    st.subheader("Neumann Boundary Conditions")

    st.write("**Left Boundary (Flux at x=0)**")
    st.latex(r'''
    \left.\frac{\partial T}{\partial x}\right|_{x=0} = q_L
    \quad \Rightarrow \quad
    T_0^{\,n+1} = T_1^{\,n+1} - q_L \, \Delta x
    ''')
    st.latex(r'''
    (1 + 2r) T_0^{\,n+1} - 2r T_1^{\,n+1} = T_0^{\,n} + 2r \, \Delta x \, q_L
    ''')

    st.write("**Right Boundary (Flux at x=L)**")
    st.latex(r'''
    \left.\frac{\partial T}{\partial x}\right|_{x=L} = q_R
    \quad \Rightarrow \quad
    T_N^{\,n+1} = T_{N-1}^{\,n+1} + q_R \, \Delta x
    ''')
    st.latex(r'''
    -2r T_{N-1}^{\,n+1} + (1 + 2r) T_N^{\,n+1} = T_N^{\,n} + 2r \, \Delta x \, q_R
    ''')

    # Kondisi Awal
    st.subheader("Initial Condition")
    st.latex(r'''
    T_i^{\,0} = T_{\text{init}}, \quad \forall i
    ''')

    # Kondisi Stabilitas
    st.subheader("Stability Condition")
    st.latex(r'''
    \text{Unconditionally stable for all } \; r = \frac{\alpha \, \Delta t}{\Delta x^2} > 0
    ''')

#----------------------------------------------------------------------------------------------------------------------

# Calculation Heat Equation
if solution == "Explicit":
    def ftcs_heat_solver (alpha, length, n, total_time, dt, T0, 
                          left_boundary, right_boundary):

        # === Discretization ===
        grid_number = n
        dx = length/n
        if dt is None:
            dt = r_target * dx**2 /alpha

        # === Stability Check ===
        r = alpha * dt / dx**2
        if r > 0.5:
            print(f"⚠️ Warning: Scheme unstable! r = {r:.3f} > 0.5")
        else:
            print(f"✅ Stable: r = {r:.3f}")

        # === Grid Setup ===
        x = np.linspace(0, length, n+1)
        nt = int(total_time / dt)
        times = np.linspace(0, total_time, nt+1)

        # === initialize Temperature ===
        U = np.zeros((nt + 1, n +1))
        U[0, :] = T0

        # === Time Marching ===
        for n in range(0, nt):
            for i in range (1, grid_number):
                U[n + 1, i] = U[n, i] + r * (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1])

            # --- Left Boundary ---
            if left_boundary == "Dirichlet":
                U[n + 1, 0] = TL
            elif left_boundary == "Neumann":
                U[n + 1, 0] = U[n + 1, 1] - qL * dx

            # --- Right Boundary ---
            if right_boundary == "Dirichlet":
                U[n + 1, -1] = TR
            elif right_boundary == "Neumann":
                U[n + 1, -1] = U[n + 1, -2] + qR * dx

        return x, times, U
    x, times, U = ftcs_heat_solver(alpha, length, n, total_time, dt, T0, 
                               left_boundary, right_boundary)    
elif solution == "Implicit":
    def implicit_heat_solver(alpha, length, n, total_time, dt, T0, 
                          left_boundary, right_boundary):
        dx = length / n
        x = np.linspace(0, length, n + 1)
        nt = int(np.ceil(total_time / dt))
        times = np.linspace(0, nt * dt, nt + 1)

        r = alpha * dt / dx**2

        # --- Construct coefficien matrix A for implicit method ---
        A = np.zeros((n + 1, n + 1))

        # Internal nodes
        for i in range (1, n):
            A[i, i-1] = -r
            A[i, i] = 1 + 2 * r
            A[i, i+1] = -r
        
        # Boundary Condition
        # Left Boundary
        if left_boundary == "Dirichlet":
            A[0, :] = 0
            A[0, 0] = 1
        else:
            A[0, 0] = 1 + 2 * r
            A[0, 1] = -2 * r
        
        # Right Boundary
        if right_boundary == "Dirichlet":
            A[-1, :] = 0
            A[-1, -1] = 1
        else:
            A[-1, -2] = -2 * r
            A[-1, -1] = 1 + 2 *r

        # --- Initialize temperature field ---
        U = np.zeros((nt + 1, n + 1))
        U[0, :] = T0

        # Apply initial Dirichlet BCs
        if left_boundary == "Dirichlet":
            U[0, 0] = TL
        if right_boundary == "Dirichlet":
            U[0, -1] = TR

        # --- Time stepping
        for m in range (nt):
            b = U[m, :].copy()

            # Adjust RHS for Neuman BCs
            if left_boundary == "Neumann":
                b[0] += 2 * r * dx * qL
            if right_boundary == "Neumann":
                b[-1] += 2 * r * dx * qR
            
            # Enforce Dirichlet BCs in RHS
            if left_boundary == "Dirichlet":
                b[0] = TL
            if right_boundary == "Dirichlet":
                b[-1] = TR
            
            # Solve implicit system
            U[m+1, :] = np.linalg.solve(A, b)
        
        return x, times, U
    x, times, U  = implicit_heat_solver(alpha, length, n, total_time, dt,
                                   T0, left_boundary, right_boundary)
#----------------------------------------------------------------------------------------------------------------------

# Visualize Temperature Distribution
st.subheader("Visualize Temperature Distribution")

# 3d Visulization
x_mesh = np.arange(U.shape[1])   # Number of spatial sections
y_mesh = np.arange(U.shape[0])   # Time steps
x_mesh, y_mesh = np.meshgrid(x_mesh, y_mesh)

fig = go.Figure(data=[go.Surface(z=U, x=x_mesh, y=y_mesh, colorscale="Magma")])

fig.update_layout(
    scene=dict(
        xaxis_title="Number of Sections",
        yaxis_title="Time Step",
        zaxis_title="Temperature (°C)",
    ),
    autosize=True,
    height=700,
)

st.plotly_chart(fig, use_container_width=True)

# Graph Visualization
df = pd.DataFrame(U.T)
df.index.name = "Section"
df = df.reset_index().melt(
    id_vars="Section", var_name="Timestep", value_name="Temperature"
)
df["Timestep"] = df["Timestep"].astype(int)

fig = px.line(
    df,
    x="Section",
    y="Temperature",
    animation_frame="Timestep",
    range_y=[df["Temperature"].min(), df["Temperature"].max()],
    markers=True,
    title="Temperature Evolution Along the Rod",
)

fig.update_layout(
    xaxis_title="Number of Sections",
    yaxis_title="Temperature (°C)",
    height=600,
    showlegend=False,
)

frame_duration = 50     # ms per frame → cepat
transition_duration = 0  # tanpa efek transisi lambat

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = transition_duration

st.plotly_chart(fig, use_container_width=True)

# Heatmap Visualization
nt = U.shape[0] 
auto_play = st.checkbox("Auto Play", value=False)
plot_placeholder = st.empty()

vmin, vmax = U.min(), U.max()

def plot_timestep(selected_timestep):
    temperature_profile = U[selected_timestep, :].reshape(1, -1)

    fig, ax = plt.subplots(figsize=(10, 2))
    img = ax.imshow(
        temperature_profile,
        cmap='magma',
        aspect='auto',
        extent=[0, U.shape[1], 0, 1],
        vmin=vmin,
        vmax=vmax
    )

    cbar = fig.colorbar(img, ax=ax, shrink=1.0, aspect=5)
    cbar.set_label('Temperature (K)')

    ax.set_xlabel('Number of Sections')
    ax.set_yticks([])
    ax.set_title(f'Temperature Distribution at Timestep {selected_timestep}')

    plot_placeholder.pyplot(fig)
    plt.close(fig)

if auto_play:
    for t in range(nt):
        plot_timestep(t)
        time.sleep(0.01)
else:
    selected_timestep = st.slider(
        "Select Timestep", min_value=0, max_value=nt - 1, value=0, step=1
    )
    plot_timestep(selected_timestep)

# Dataframe
st.dataframe(U)
