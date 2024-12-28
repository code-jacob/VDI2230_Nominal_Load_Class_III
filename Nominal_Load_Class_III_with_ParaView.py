"""
Created on Fri Jul 14 10:52:49 2023
Author: Jakub Trušina
Name: Nominal_Load_Class_III_with_ParaView.py
"""

inp = [ "SIGM_from_ParaView_side_TOP.txt", "SIGM_from_ParaView_side_BOTTOM.txt" ]            # File 
result_name = "STA_NL__"


reference_point_1 = 50
reference_point_2 = 150

# activate - 1 , deactivate - 0
axisym = 0                          # axisymmetric study X - radial direction , Y - axial direction 

# =============================================================================
# VDI 2230 - Part 2, Model class III - Additional bolt load from external operating loads
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

it = 0
plt.close("all")
for item in inp:
    it = it + 1
    df = pd.read_csv(item)
    collist = ["Points_0","Points_1","Points_2","arc_length",result_name+"SIGM_NOEU_0",result_name+"SIGM_NOEU_1", result_name+"SIGM_NOEU_2", result_name+"SIGM_NOEU_3", result_name+"SIGM_NOEU_4", result_name+"SIGM_NOEU_5" ]
    df = df.reindex(columns=collist)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 10)
    print(df)
    data = df.to_numpy()
    
    point_1 = [ data[0,0] , data[0,1] , data[0,2] ]
    point_2 = [ data[-1,0] , data[-1,1] , data[-1,2] ]
    
    dx = point_2[0]-point_1[0]
    dy = point_2[1]-point_1[1]
    dz = point_2[2]-point_1[2]
    L12 = np.sqrt(dx**2+dy**2+dz**2)
    
    x = data[:,3]
    SX = data[:,4]
    SY = data[:,5]
    SZ = data[:,6]
    SXY = data[:,7]
    
    if axisym == 1:
        print("\n Axisymmetric Problem")
        data = data[:, 0:8]
        SXZ = np.zeros(len(data[:,0]))
        SYZ = np.zeros(len(data[:,0]))
        data = (np.concatenate((data.transpose(),  [np.zeros(len(x)), np.zeros(len(x))] ))).transpose()
    else:
        print("\n 3D Problem")
        SXZ = data[:,8]
        SYZ = data[:,9]
    
    Stress = np.array([SX,SY,SZ,SXY,SXZ,SYZ])
    
    h = x[-1]
    print(" Line Length = ", h); print(" Number of Points = ", len(x))
    print( " reference point 1 for linear extrapolation = ", str("%.2f"%reference_point_1) )
    print( " reference point 2 for linear extrapolation = ", str("%.2f"%reference_point_2) )
    
    def Tresca_stress(s11,s22,s33,s12,s13,s23):
        tresca_stress = np.zeros(len(s11))
        sigma_1 = np.zeros(len(s11)); sigma_2 = np.zeros(len(s11)); sigma_3 = np.zeros(len(s11))
        for i in range(0,len(s11)):
            sigma = np.array([ [s11[i], s12[i], s13[i] ],
                               [s12[i], s22[i], s23[i] ],
                               [s13[i], s23[i], s33[i] ] ])
            v, n = np.linalg.eig(sigma)
            v = np.sort(v)
            sigma_1[i] = v[2]
            sigma_2[i] = v[1]
            sigma_3[i] = v[0]        
            tresca_stress[i] = sigma_1[i] - sigma_3[i]
            # tresca_stress[i] = max( abs(sigma_1[i]-sigma_2[i]), abs(sigma_2[i]-sigma_3[i]), abs(sigma_3[i]-sigma_1[i]) ) 
        return tresca_stress, sigma_1, sigma_2, sigma_3
    TRESCA, sigma_1, sigma_2, sigma_3 = Tresca_stress(SX,SY,SZ,SXY,SXZ,SYZ)
    Principal_Stress = np.array([sigma_1,sigma_2,sigma_3])
    
    def vMis_stress(s1,s2,s3):
        stress = np.sqrt( 0.5*((s1-s2)**2+(s2-s3)**2+(s3-s1)**2))
        return stress
    VMIS = vMis_stress(sigma_1,sigma_2,sigma_3)
    
    def Approximation_linear(data,ref_point_1,ref_point_2, dx , xx):
        S_1 = np.zeros((6)); S_2 = np.zeros((6))
        y1 = np.zeros((6)) ; y2 = np.zeros((6)) 
        c0 = np.zeros((6)) ; c1 = np.zeros((6)) ; y_lin = np.zeros((6,len(xx)))
        for ii in range(0,6):
            S_1[ii] = np.interp(ref_point_1, dx, data[:,ii+4])
            S_2[ii] = np.interp(ref_point_2, dx, data[:,ii+4])    
    
            x1 = ref_point_1 ; y1[ii] = S_1[ii]
            x2 = ref_point_2 ; y2[ii] = S_2[ii]
    
            A = np.array([ [x1, 1],
                           [x2, 1] ])
            b = np.array( [ y1[ii], y2[ii]] )
            c = np.linalg.solve(A,b)
            c0[ii] = c[0] ; c1[ii] = c[1]
    
            def linear_fun(bb,cc,x_val):
                y = bb*x_val + cc
                return y
            y_lin[ii] = linear_fun(c0[ii],c1[ii],xx)
        return y_lin , S_1 , S_2 
    
    x_lin = np.linspace(0,h,100)
    f_lin, S1, S2 = Approximation_linear(data,reference_point_1,reference_point_2, x, x_lin)

    TRESCA_lin, sigma_1_lin, sigma_2_lin, sigma_3_lin = Tresca_stress( f_lin[0],f_lin[1],f_lin[2],f_lin[3],f_lin[4],f_lin[5] )
    TRESCA_S1, sigma_1_S1, sigma_2_S1, sigma_3_S1 = Tresca_stress( [S1[0]],[S1[1]],[S1[2]],[S1[3]],[S1[4]],[S1[5]] )
    TRESCA_S2, sigma_1_S2, sigma_2_S2, sigma_3_S2 = Tresca_stress( [S2[0]],[S2[1]],[S2[2]],[S2[3]],[S2[4]],[S2[5]] )
    VMIS_lin = vMis_stress(sigma_1_lin,sigma_2_lin,sigma_3_lin)
    VMIS_S1 = vMis_stress( sigma_1_S1, sigma_2_S1, sigma_3_S1 )
    VMIS_S2 = vMis_stress( sigma_1_S2, sigma_2_S2, sigma_3_S2 )

    if it == 1:
        fig = plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
        fig.canvas.manager.set_window_title("Nominal Bolt Loading - Equivalent Stresses")
        # plt.axes( facecolor='#DBEDFD')
        plt.title( "Nominal Bolt Loadings - Equivalent Stresses" , fontsize= 20)        
        
        VMIS_TOP = VMIS
        VMIS_S1_TOP = VMIS_S1 ; VMIS_S2_TOP = VMIS_S2
        VMIS_lin_TOP = VMIS_lin
        
        plt.plot(x, VMIS_TOP,'r-', label= 'von Mises Stress TOP', linewidth= 3 )
        plt.plot(x, VMIS_TOP, 's', markersize= 3, color= 'r' )
        plt.plot(x_lin, VMIS_lin_TOP,'r--', label= 'Linear Approximation TOP', linewidth= 3 )
        plt.scatter(x_lin[0],VMIS_lin_TOP[0],color='r',marker="x",s=40, label='Point 1 = ' + str("%.0f"%(VMIS_lin_TOP[0])))
        plt.scatter(x_lin[-1],VMIS_lin_TOP[-1],color='r',marker="x",s=40, label='Point 2 = ' + str("%.0f"%(VMIS_lin_TOP[-1])))
        plt.plot(reference_point_1, VMIS_S1_TOP, 'o', markersize= 7, color= 'r' , label='Reference point' )
        plt.text(x_lin[0], VMIS_lin_TOP[0], str("%.0f"%(VMIS_lin_TOP[0])) , fontsize= 16,color="r") 
        plt.text(x_lin[-1], VMIS_lin_TOP[-1], str("%.0f"%(VMIS_lin_TOP[-1])) , fontsize= 16,color="r") 
        plt.plot(reference_point_2, VMIS_S2_TOP, 'o', markersize= 7, color= 'r'  )
        
    else: 
        VMIS_BOTTOM = VMIS
        VMIS_S1_BOTTOM = VMIS_S1 ; VMIS_S2_BOTTOM = VMIS_S2
        VMIS_lin_BOTTOM = VMIS_lin
        
        plt.plot(x, VMIS_BOTTOM,'b-', label= 'von Mises Stress BOTTOM', linewidth= 3 )
        plt.plot(x, VMIS_BOTTOM, 's', markersize= 3, color= 'b' )
        plt.plot(x_lin, VMIS_lin_BOTTOM,'b--', label= 'Linear Approximation BOTTOM', linewidth= 3 )
        plt.scatter(x_lin[0],VMIS_lin_BOTTOM[0],color='b',marker="x",s=40, label='Point 1 = ' + str("%.0f"%(VMIS_lin_BOTTOM[0])))
        plt.scatter(x_lin[-1],VMIS_lin_BOTTOM[-1],color='b',marker="x",s=40, label='Point 2 = ' + str("%.0f"%(VMIS_lin_BOTTOM[-1])))
        plt.text(x_lin[0], VMIS_lin_BOTTOM[0], str("%.0f"%(VMIS_lin_BOTTOM[0])) , fontsize= 16,color="b") 
        plt.text(x_lin[-1], VMIS_lin_BOTTOM[-1], str("%.0f"%(VMIS_lin_BOTTOM[-1])), fontsize= 16,color="b") 
        plt.plot(reference_point_1, VMIS_S1_BOTTOM, 'o', markersize= 7, color= 'b', label='Reference point'  )
        plt.plot(reference_point_2, VMIS_S2_BOTTOM, 'o', markersize= 7, color= 'b'  )
        
        VMIS_lin_Membrane = (VMIS_lin_TOP + VMIS_lin_BOTTOM)/2
        plt.plot(x_lin, VMIS_lin_Membrane,'k--', label= 'Linear Approximation Membrane', linewidth= 3 )
        plt.scatter(x_lin[0],VMIS_lin_Membrane[0],color='k',marker="x",s=40, label='Point 1 = ' + str("%.0f"%(VMIS_lin_Membrane[0])))
        plt.scatter(x_lin[-1],VMIS_lin_Membrane[-1],color='k',marker="x",s=40, label='Point 2 = ' + str("%.0f"%(VMIS_lin_Membrane[-1])))
        plt.text(x_lin[0], VMIS_lin_Membrane[0], str("%.0f"%(VMIS_lin_Membrane[0])) , fontsize= 16,color="k") 
        plt.text(x_lin[-1], VMIS_lin_Membrane[-1], str("%.0f"%(VMIS_lin_Membrane[-1])) , fontsize= 16,color="k") 

        VMIS_lin_Bending = abs((VMIS_lin_TOP - VMIS_lin_BOTTOM)/2)
        plt.plot(x_lin, VMIS_lin_Bending,'C0--', label= 'Linear Approximation Bending', linewidth= 3 )
        plt.scatter(x_lin[0],VMIS_lin_Bending[0],color='C0',marker="x",s=40, label='Point 1 = ' + str("%.0f"%(VMIS_lin_Bending[0])))
        plt.scatter(x_lin[-1],VMIS_lin_Bending[-1],color='C0',marker="x",s=40, label='Point 2 = ' + str("%.0f"%(VMIS_lin_Bending[-1])))
        plt.text(x_lin[0], VMIS_lin_Bending[0],  str("%.0f"%(VMIS_lin_Bending[0]))  , fontsize= 16,color="C0") 
        plt.text(x_lin[-1], VMIS_lin_Bending[-1], str("%.0f"%(VMIS_lin_Bending[-1])) , fontsize= 16,color="C0")        
    
    
    plt.ylabel(' $\sigma$ $[MPa]$ ', fontsize = 15)
    plt.xlabel(r' Distance [mm]', fontsize = 15)
    plt.legend(loc='best',  shadow= True,  ncol=2, fontsize= 14)
    plt.grid(linestyle= '-', linewidth= 1)
    plt.tight_layout()
    # plt.yscale('log')
    plt.show()
    





