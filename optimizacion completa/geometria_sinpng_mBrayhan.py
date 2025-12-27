
import numpy as np
import meep as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from meep.materials import Cu




def ejecutar_simulacion(params):
    a, b, d, h, c = params

    # Banda de la fuente 
    # pueden ser Banda 2: 67 - 90 Ghz, Banda3: 84 -116 GHz, Banda 2+3: 67-116
    # dejando c=1 y unidades milimetricas como unidad principal es decir j=1 es 1mm
    # para convertir primero comvierto la frecuencia en GHZ en longitud de onda en mm, usar \lamda = 300/f(GHz) (mm), luego f = 1/lambda, es decir f(UNIDADES MEEP) = f(GHz)/300 
    # banda 2: 2.23 - 3, banda 3: 2.8 - 3.87, banda 2+3: 2.23 - 3.87  

    fminima = 75
    fmaxima = 110

    fminima = fminima/ 300 # en milimetros
    fmaxima = fmaxima/ 300 # en milimetros

    fcentral  = (fminima + fmaxima)/2
    df = np.abs(fminima - fmaxima)

    n = 8  # número de brazos del híbrido en H
    L = (n-1)*(c+h) + h # longitud total del híbrido en H
    Nlamda = int(round(L*fminima) + 1)

    # Conversión a unidades internas de Meep (1 unit = 1 mm)
    
    cell_x = (Nlamda)*1/fminima + 1.5#d+b+3 * um
    cell_y = 2*b + d + 1.5 
    cell_z = a +1.5 #a+2 * um
    resolution = 40 # pts/mm Resoluciones bajas dan errores de INf o Nan en los ca,pos al pareceser necesita una alta densidad ed puntos por milimetro

 

    nfreq = 100
    # Fuente ubicada en uno de los extremos de la región 


    sources = [
        mp.Source(
            mp.ContinuousSource(frequency=fcentral, fwidth = df), component=mp.Bz,size=(0, b,a) ,center=mp.Vector3(-Nlamda/(2*fminima),(b+d)/2,  0),
        )
    ]

    # monitores de flujo 

    # lista de posiciones o regiones donde pones monitores
    regions = [
        mp.FluxRegion(center=mp.Vector3(  -Nlamda/(2*fminima) +0.1, (b+d)/2, 0), size=mp.Vector3 (0, b, a)), # Monitor de la fuente     
        mp.FluxRegion(center=mp.Vector3(  Nlamda/(2*fminima), (b+d)/2, 0), size=mp.Vector3 (0, b, a)), # puerto S2
        mp.FluxRegion(center=mp.Vector3(  Nlamda/(2*fminima), -(b+d)/2, 0), size=mp.Vector3 (0, b, a)), # puerto S3
    ]

    def geometria(a,b,c,L,d,h,n):
        L = 10.0
        n = 8
        if n%2 == 0:
            # ————————————————————————————————
            # Definición de la geometría en "H"  para numero de brazos pares 
            m = int(n/2)
            mBrach = h*(m + 1) + c*(1/2 + m)
            
            if mBrach > L: # verificación de errores por numero excesivo de brazos
                print("numero de Brazos incorrecto, no capen en la altura de la guia dada")
                return
            
            geometry = [
                mp.Block(size = mp.Vector3( Nlamda/(fminima)+1,cell_y,cell_z), # Cubo de Metal al rededor del hibrido 
                            center = mp.Vector3(0,0,0),
                            material = mp.metal),
                mp.Block(size=mp.Vector3(Nlamda/(fminima)+1, b, a), # Conductos principales
                            center=mp.Vector3( 0, -(b+d)/2, 0), 
                            material=mp.air),
                mp.Block(size=mp.Vector3( Nlamda/(fminima)+1, b,  a),
                            center=mp.Vector3(0, (b+d)/2, 0),
                            material=mp.air)] + [ 
                mp.Block(size=mp.Vector3(h, d, a), 
                        center=mp.Vector3( -(1+2*i)*(h+c)/2, 0, 0), # Brazos extra 
                        material=mp.air) for i in range(m) ] + [
                mp.Block(size=mp.Vector3( h, d, a), 
                        center=mp.Vector3((1+2*i)*(h+c)/2 , 0,  0), 
                        material=mp.air) for i in range(m)]
                        
            return geometry
            
            
        else:
            # ————————————————————————————————
            # Definición de la geometría en "H" para numero de brazos impares
            
            m = int((n-1)/2)
            mBrach = c*m + h*(m + 1/2)
            
            if mBrach > L: # verificación de errores por numero excesivo de brazos
                print("numero de Brazos incorrecto, no capen en la altura de la guia dada")
                return
                
            geometry = [
                mp.Block(size = mp.Vector3(cell_x,cell_y,cell_z),
                            center = mp.Vector3(0,0,0),
                            material = mp.metal), #Recubrimiento, Caja hecha de por ahora Cobre (esta puesta en debate)
                mp.Block(size=mp.Vector3( cell_x, b, a),
                        center=mp.Vector3(0, -(b+d)/2,  0),
                        material=mp.air),
                mp.Block(size=mp.Vector3(cell_x, b, a),   #revisar-------------------OJO
                        center=mp.Vector3( 0, (b+d)/2, 0),
                        material=mp.air),
                mp.Block(size=mp.Vector3(h, d,  a),
                        center=mp.Vector3(0, 0, 0),
                        material=mp.air)] + [
                mp.Block(size=mp.Vector3(h,d, a),
                        center=mp.Vector3((-h-c)*(i+1), 0,  0),
                        material=mp.air) for i in range(m)] + [
                mp.Block(size=mp.Vector3( h, d, a),
                        center=mp.Vector3((h+c)*(i+1) , 0,  0),
                        material=mp.air)
            for i in range(m)]
            return geometry


    #SIMULACION CON GEOMETRIA

    geometry = geometria(a,b,c,L,d,h,n) 
    
    
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y, cell_z),
        geometry=geometry,
        boundary_layers=[mp.PML(0.25)],
        sources=sources,
        resolution=resolution,
    )

    """ Función para dibujar un bloque 3D """
    """ 
    def draw_block(ax, center, size, color='gray', alpha=0.5):
        cx, cy, cz = center.x, center.y, center.z
        sx, sy, sz = size.x/2, size.y/2, size.z/2
        
        # Coordenadas de los vértices del cubo
        vertices = [
            [cx - sx, cy - sy, cz - sz],
            [cx + sx, cy - sy, cz - sz],
            [cx + sx, cy + sy, cz - sz],
            [cx - sx, cy + sy, cz - sz],
            [cx - sx, cy - sy, cz + sz],
            [cx + sx, cy - sy, cz + sz],
            [cx + sx, cy + sy, cz + sz],
            [cx - sx, cy + sy, cz + sz]
        ]
        
        # Caras del cubo
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        ]
        
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))
    """
    # ————————————————————————————————
    # Visualización 3D de la geometría
    '''
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for obj in geometry:
        if isinstance(obj, mp.Block):
            draw_block(ax, obj.center, obj.size)

    # Ajustamos título y etiquetas
    ax.set_title("Geometría 3D del híbrido en 'H'")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # — Ajuste de límites para que el cubo de ejes envuelva toda la geometría —
    #margin = 0.2  # 20% de margen extra
    #half_x = cell_x / 2
    #half_y = cell_y / 2
    #half_z = a / 2

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)

    plt.tight_layout()
    plt.savefig("geometria_3D.png", dpi=200)

    plt.close()
'''


    flux_monitors = []
    for reg in regions:
        flux_monitors.append( sim.add_flux(fcentral, df, nfreq, reg) )

    # vista 2D de la geometria del hibrido 
    fig = plt.figure(figsize=(6,6))
    sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0)))
    plt.title("Geometría en 'H' del híbrido en cuadratura")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.axis("equal")
    plt.savefig("geometria_2D.png", dpi=200)
    plt.close()

    

  
    



    # ahora ejecutamos la simulación (mantén tu sim.run original)
    global fluxtime
    fluxtime = []

    def print_flux(sim) :
        fluxtime.append([ mp.get_fluxes(fm) for fm in flux_monitors ])

    sim.run(mp.at_every(1/fminima, print_flux),                 # cada 50 unidades de tiempo
            until=15*1/fminima)



    plt.figure(figsize=(10,10))
    sim.plot2D(output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(cell_x, cell_y, 0)),fields=mp.Bz) # cambiar entre Ey y Bz para ver lso diferentes campos 
    plt.title("Vista 2D del campo Elcectrico en su componente Ey")
    plt.savefig("vista2Dcampoelectrico.png", dpi=200)
    plt.close()

    # ANÁLISIS DE LA DIFERENCIA DE FASE ENTRE LOS PUERTOS 2 Y 3

    


    # --- EXTRAEMOS Y PLOTAMOS FLUJO VS LONGITUD DE ONDA ---
    # obtenemos las frecuencias de muestreo
    freqs = np.array(mp.get_flux_freqs(flux_monitors[0]))*300  # en GHz

    # recogemos los flujos de cada monitor
    naflux =np.array(fluxtime)

    flujofunete  = naflux[14-(Nlamda+1),0,:] 
    flujos2 = naflux[14,1,:] 
    flujos3 = naflux[14,2,:] # dato a usar en la optimización


    #EXTRACCIÓN DE PARÁMETROS S


    # S-PARÁMETROS
    S21 = flujos2 / flujofunete
    S31 = flujos3 / flujofunete
    S11 = 0 #Asumiendo que no hay reflexión en el puerto 1 

    # Magnitud en dB
    S21_dB = 10*np.log10(S21)
    S31_dB = 10*np.log10(S31)
    #S11_dB = 10*np.log10(np.abs(S11))

    # desfase en grados
    ang = np.arctan(S21_dB, S31_dB) * (180/np.pi)

    plt.figure(figsize=(10,6))
    plt.plot(freqs, ang, label='Diferencia de fase ∠(S12/S13)')
    plt.plot(freqs, [90]*len(freqs), "k--" , label=' Diferencia Ideal = 90°')
    plt.xlabel('Frecuencia (GHz)')
    plt.ylabel('Diferencia de Fase (grados)')
    plt.title('Diferencia de Fase entre Puertos del Híbrido en H')
    plt.legend()
    plt.grid()
    plt.savefig("Desfase.png", dpi=300)
    plt.close()


    #  AMPLITUD
    deltaA = np.abs(S21_dB/ S31_dB)
    plt.figure(figsize=(10,6))
    plt.plot(freqs, deltaA, label='Relación de amplitud |S12/S13|')
    plt.plot(freqs, [1]*len(freqs), "k--" ,label=' Relación Ideal = 1')
    plt.xlabel('Frecuencia (GHz)')
    plt.ylabel('Relación de Amplitud')
    plt.title('Relación de Amplitud entre Puertos del Híbrido en H')
    plt.legend()
    plt.grid()
    plt.savefig("Amplitud.png", dpi=300)
    plt.close()
#FLUJOS


#Flujos no dB
    plt.figure(figsize=(8,6))
   
    plt.plot(np.array(freqs)*300, flujofunete, label='Puerto 1: Incidente')
    plt.plot(np.array(freqs)*300, flujos2, label='Puerto salida 2')
    plt.plot(np.array(freqs)*300, flujos3, label='Puerto salida 3 ')
    plt.xlabel('Frecuencia Ghz')
    plt.ylabel('Flujo (unidad Meep)')
    plt.title('Espectro de Potencia en cada puerto')
    plt.legend()
    plt.gca()  # opcional: λ creciente a la izquierda
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("espectro.png", dpi=300)
    plt.close()




    #Parámetros S
    plt.figure(figsize=(8,6))

    plt.plot(np.array(freqs)*300, S21_dB, label='S21 (dB)')
    plt.plot(np.array(freqs)*300, S31_dB, label='S31 (dB)')
    plt.plot(np.array(freqs)*300, -3*np.ones_like(freqs), 'k--', label='-3 dB')

    plt.xlabel('Frecuencia (Ghz)')
    plt.ylabel('Parámetros de dispersión (dB)') 
    plt.title('Parámetros S del híbrido en H')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sparams.png", dpi=300)
    plt.close()

 
    
    print(f"parametros=(a={a}, b={b}, L={L}, d={d}, h={h}, c={c}, n={n}")
    
    parametros = freqs, S21_dB, S31_dB, ang, deltaA
    return parametros








    
parametros_de_prueba =[2.54, 1.27, 0.74, 0.2866, 0.74]
ejecutar_simulacion(parametros_de_prueba)
