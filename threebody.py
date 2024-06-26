#==================================================================
#            Simulation of the Three Bodies Problem
#==================================================================

import numpy as np 
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

#==================================================================
#                       Initial Values
#==================================================================

#Time (Evolution time,time increasment)
T,dt=(150,0.1)
#Visualize bodies trajectories
plot_orbits=False
#Save animation
save_ani=False

#BODY 1
name1 = "Body 1"
mass1 = 0.4
x0_1  = np.array([0,0,0])
v0_1  = np.array([-0.01,0.1,0.01])

#BODY 2
name2 = "Body 2"
mass2 = 0.3
x0_2  = np.array([3,0,0])
v0_2  = np.array([0.01,-0.1,-0.01])

#BODY 3
name3 = "Body 3"
mass3 = 1-mass1-mass2
x0_3  = np.array([2,1,-1])
v0_3  = -(mass1*v0_1+mass2*v0_2)/mass3








#Printing format
class Colors:
    ENDC   = '\033[0m'
    BOLD   = '\033[1m'
    GREEN  = '\033[1;32m'
    BLUE   = '\033[1;34m'
    YELLOW = '\033[1;33m'
    CYAN   = '\033[1;36m'
    RED    = '\033[1;31m'






# Define a body class
class Body:
    def __init__(self,mass=1,x=np.zeros(3),v=np.zeros(3),name='Body'):
        #Check if the vectors has the right number of dimensions (3)
        if len(x)!=3 or len(v)!=3:
            raise ValueError(f'Unexpected lenght ( {len(x)} , {len(v)} ) for {name}!')

        #Mass
        self.mass=mass
        #Coordinates (Position + Velocity)
        self._coord=np.append(x,v)
        #Body's name
        self.name=name
    
    #Get position vector
    def get_position(self):
        return self._coord[:3]
    
    #Get velocity vector
    def get_velocity(self):
        return self._coord[3:]
    
    #Set coordinates
    def set_coords(self,x):
        if len(x)!=6:
            raise ValueError(f'Unexpected lenght ( {len(x)} , {len(v)} ) for {name}!')
        self._coord=np.copy(x)

    #Get coordinates
    def get_coords(self):
        return self._coord
    
    #Get any coordinate
    def __getitem__(self,index):
        return self._coord[index]
    
    #Print values
    def __str__(self):
        line= Colors.YELLOW+self.name+' :\n'+Colors.ENDC
        line+=f'\t{Colors.GREEN}Mass{Colors.ENDC}      = {self.mass:1.2f}\n'
        line+=f'\t{Colors.GREEN}Position{Colors.ENDC}  = {self.get_position()}\n'
        line+=f'\t{Colors.GREEN}Velocity{Colors.ENDC}  = {self.get_velocity()}\n'

        return line
    
    #Compute the gravitational accelleration of another body
    def __call__(self,body):
        #Displacement of the other body
        Dx = body.get_position()-self.get_position()
        #Distance^3 of the other body
        distance3 = np.sqrt(np.sum(Dx**2))**3

        return body.mass*Dx/distance3











#==================================================================
#                      Global Variables
#==================================================================

#Bodies
b1=Body(
    name=name1,
    mass=mass1,
    x=x0_1,
    v=v0_1
)
b2=Body(
    name=name2,
    mass=mass2,
    x=x0_2,
    v=v0_2
)
b3=Body(
    name=name3,
    mass=mass3,
    x=x0_3,
    v=v0_3
)


#==================================================================
#                         Animation
#==================================================================
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.grid()

#Body 1
color1='royalblue'
point1,=ax.plot([],[],[],marker='o',linestyle='',color=color1,label=b1.name)
line1, =ax.plot([],[],[],color=color1)
x1=np.array([b1.get_position()])

#Body 2
color2='orange'
point2,=ax.plot([],[],[],marker='o',linestyle='',color=color2,label=b2.name)
line2, =ax.plot([],[],[],color=color2)
x2=np.array([b2.get_position()])
#Body 3
color3='magenta'
point3,=ax.plot([],[],[],marker='o',linestyle='',color=color3,label=b3.name)
line3, =ax.plot([],[],[],color=color3)
x3=np.array([b3.get_position()])








#Initialize animation
def init():
    #Set axis limit
    xmin=min([min(x1[0]),min(x2[0]),min(x3[0])])
    xmax=max([max(x1[0]),max(x2[0]),max(x3[0])])

    ymin=min([min(x1[1]),min(x2[1]),min(x3[1])])
    ymax=max([max(x1[1]),max(x2[1]),max(x3[1])])

    zmin=min([min(x1[2]),min(x2[2]),min(x3[2])])
    zmax=max([max(x1[2]),max(x2[2]),max(x3[2])])

    side_x=(xmax-xmin)*0.05
    side_y=(ymax-ymin)*0.05
    side_z=(zmax-zmin)*0.05
    
    ax.set_xlim(xmin-side_x,xmax+side_x)
    ax.set_ylim(ymin-side_y,ymax+side_y)
    ax.set_zlim(zmin-side_z,zmax+side_z)

    #Initiate the points
    point1.set_data([x1[0, 0]], [x1[0, 1]])
    point1.set_3d_properties([x1[0, 2]])

    point2.set_data([x2[0,0]],[x2[0,1]])
    point2.set_3d_properties([x2[0,2]])

    point3.set_data([x3[0,0]],[x3[0,1]])
    point3.set_3d_properties([x3[0,2]])

    #Add legend
    ax.legend()
    return point1,point2,point3


def animation(i):
    #Find new coordinates
    x_1=x1[0,i]
    y_1=x1[1,i]
    z_1=x1[2,i]

    x_2=x2[0,i]
    y_2=x2[1,i]
    z_2=x2[2,i]

    x_3=x3[0,i]
    y_3=x3[1,i]
    z_3=x3[2,i]

    #Update the points
    point1.set_data([x_1],[y_1])
    point1.set_3d_properties([z_1])

    point2.set_data([x_2],[y_2])
    point2.set_3d_properties([z_2])

    point3.set_data([x_3],[y_3])
    point3.set_3d_properties([z_3])

    return point1,point2,point3












#System of differential equations
def system_ode(t,y,progress):
    b1.set_coords(y[:6])
    b2.set_coords(y[6:12])
    b3.set_coords(y[12:])
    progress.update(t-progress.n)
    return np.concatenate((y[3:6],b1(b2)+b1(b3),y[9:12],b2(b1)+b2(b3),y[15:],b3(b1)+b3(b2)))



#Compute bodyes trajectories
def evolution(N):
    print('Computing Trejectories',end='...')

    y0=np.concatenate((b1.get_coords(),b2.get_coords(),b3.get_coords()))


    #Loading bar estetic
    desc       = Colors.BLUE+'Computing'+Colors.ENDC
    bar_format = '{desc}:|{bar}| {n:3.2f} {unit} /{total} {unit} | [ {elapsed} < {remaining} , {rate_fmt} ]'
    with tqdm(total=T,desc=desc,unit='s',bar_format=bar_format,colour='cyan') as pbar:
        x=solve_ivp(
            system_ode,
            (0,T),y0,
            t_eval=np.linspace(0,T,num=N,endpoint=True),
            method='LSODA',
            args=(pbar,)
        )

    assert x.success

    return x.y


#==================================================================
#                            Main
#==================================================================
def main():
    print(Colors.CYAN+'='*50)
    print(f"\t\tThree Bodies Problem")
    print('='*50+Colors.ENDC)

    print(f'{Colors.RED}\n\tT_MAX ={Colors.ENDC} {T:4d} , {Colors.RED} T_STEP ={Colors.ENDC} {dt}\n')

    print(b1)
    print(b2)
    print(b3)

    #Number of iteration
    N=int(T/dt)

    #Get trajectories
    x=evolution(N)
    appo=np.reshape(x,(3,6,len(x[0])))

    global x1,x2,x3
    x1=np.copy(appo[0])
    x2=np.copy(appo[1])
    x3=np.copy(appo[2])
    
    #Plots orbits
    print(f'{Colors.RED}Tajectories Lines = {Colors.ENDC}{plot_orbits}')
    if plot_orbits:
        line1.set_data(x1[0],x1[1])
        line1.set_3d_properties(x1[2])

        line2.set_data(x2[0],x2[1])
        line2.set_3d_properties(x2[2])

        line3.set_data(x3[0],x3[1])
        line3.set_3d_properties(x3[2])


    #Animation
    ani = FuncAnimation(fig, animation, init_func=init, frames=N, interval=10, blit=True)
    if save_ani :
        ani.save('threebody.gif', writer='imagemagick')
    plt.show()

if __name__=='__main__':
    main()