# Numerical Method and Groupping
# --------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED PACKAGES
# --------------------------------------------------------------------------------------------------------
import numpy as np
import math

# --------------------------------------------------------------------------------------------------------
# Numerical Integrate
# --------------------------------------------------------------------------------------------------------

class NumericalIntegrate:
    def __init__(self):
        ''' 
        This Class contains several numerical integrate method which are:
            - Left Riemann technique
            - Right Riemann technique
            - Trapezoidal technique
            - Trapezoidal upper lower technique
            - Simpson13 technique
            - Simpson38 technique
            - Auto Simpson technique
        '''
        pass

    def __input_check(self, x, y):
        # Init variables x and y
        if type(x) is not np.ndarray:
            x = np.array(x)  
        if type(y) is not np.ndarray:
            y = np.array(y)   
        #  Check Dimension
        if x.ndim != 1 and y.ndim != 1:
            raise ValueError("Either x or y is not a 1-D array")
        # Check len
        if len(x) != len(y):
            raise ValueError("Array x and y must have the same size.")
        return x, y

    def lriemann(self, x, y):
        '''Numerical integration using left Riemann technique.'''
        x, y = self.__input_check(x, y)
        subinterval =  np.delete(x,0) - np.delete(x,-1)
        sub_area = subinterval *  np.delete(y,-1)
        area = sub_area.sum()
        return area

    def rriemann(self, x, y):
        '''Numerical integration using right Riemann technique.'''
        x, y = self.__input_check(x, y)
        subinterval =  np.delete(x,0) - np.delete(x,-1)
        sub_area = subinterval *  np.delete(y,0)
        area = sub_area.sum()
        return area

    def trapezoidal(self, x, y):
        '''Numerical integration using Trapezoidal technique.'''
        x, y = self.__input_check(x, y)
        subinterval =  np.delete(x,0) - np.delete(x,-1)
        sub_area = 0.5 * subinterval * (np.delete(y,0) + np.delete(y,-1))
        area = sub_area.sum()
        return area

    def trapezoidal_upper_lower(self, x, y):
        '''# Numerical integration using Trapezoidal technique (Retrun upper and lower area seperately)'''
        x, y = self.__input_check(x, y)
        subinterval =  np.delete(x,0) - np.delete(x,-1)
        sub_area = 0.5 * subinterval * (np.delete(y,0) + np.delete(y,-1))
        upper_area = sub_area[sub_area>0].sum()
        lower_area = sub_area[sub_area<0].sum()
        return upper_area, lower_area

    def simpson13(self, x, y):
        '''# Numerical integration using Simpsom 1-3 technique (Subintervals (n) is divisible by 2).'''
        x, y = self.__input_check(x, y)
        if (len(x)-1) % 2 != 0:
            raise ValueError("Subintervals (n) is not divisible by 2")

        h = ((x[-1] - x[0])/(len(x)-1))
        mid_area = (y[2:-1:2]*2).sum() + (y[1:-1:2]*4).sum()
        area = h * (mid_area + y[0] + y[-1])/3
        return area

    def simpson38(self, x, y):
        '''Numerical integrate Using Simpsom 3-8 technique (Subintervals (n) is divisible by 3).'''
        x, y = self.__input_check(x, y)
        if (len(x)-1) % 3 != 0:
            raise ValueError("Subintervals (n) is not divisible by 3")

        h = ((x[-1] - x[0])/(len(x)-1))
        mid_area = (y[3:-1:3]*2).sum() + (y[1:-1:3]*3).sum() + (y[2:-1:3]*3).sum()
        area = h * 3 * (mid_area + y[0] + y[-1])/8
        return area

    def autosimpson(self, x, y):
        '''Auto Simpson (Automatically select between Simpson 1-3 and Simpson 3-8 or using both)'''
        x, y = self.__input_check(x, y)
        if (len(x)-1) % 2 == 0:
            area = self.simpson13(x,y)
        elif (len(x)-1) % 3 == 0:
            area = self.simpson38(x,y)
        else:  
            area = self.simpson13(x[:-3],y[:-3]) + self.simpson38(x[-4:],y[-4:])
        return area

# --------------------------------------------------------------------------------------------------------
# Numerical Root Finding
# Reference: https://en.wikipedia.org/wiki/Root-finding_algorithm
# --------------------------------------------------------------------------------------------------------

class NumericalRootFinding:
    ''' 
    This Class contains several numerical root finding method which are:
        - Regula Falsi Method (Method of false position)
        - Bisection Method
        - Secant Method

    '''
    def __init__(self):
        pass

    def regula_falsi(self, func, a, b, max_it=1000, tolerance=0.0001):
        '''
        Regula Falsi Method (Method of false position).
        Ref https://en.wikipedia.org/wiki/Regula_falsi
        '''
        if func(a) * func(b) >= 0:
            raise ValueError("The Solution is not in ths range.")

        # Initial Variable
        ai = a
        bi = b
        fai = func(a)
        fbi = func(b)
        
        for i in range(max_it):

            xi = (ai*fbi - bi*fai)/(fbi-fai)
            fxi = func(xi)
            error = abs(fxi)

            if fai * fxi > 0:
                ai = xi
                fai = fxi
            elif fbi * fxi > 0:
                bi = xi
                fbi = fxi
            else:
                return xi
            
            # Check Error Tolerance
            if error <= tolerance:
                return xi

        print('Warining: Reach maximum itteration!')    
        return xi 

    def bisection(self, func, a, b, max_it=1000, tolerance=0.0001):
        '''
        Bisection Method.
        Ref https://en.wikipedia.org/wiki/Bisection_method
        '''

        if func(a) * func(b) >= 0:
            raise ValueError("The Solution is not in ths range.")

        for i in range(max_it):
            m = (a + b)/2
            fm = func(m)
            error = abs(fm)
            if func(a) * fm < 0:
                b = m
            elif fm * func(b) < 0:
                a = m
            else:
                return m
            if error <= tolerance:
                return m

        print('Warining: Reach maximum itteration!')    
        return m            
        
    def secant(self, func, a, b, max_it=1000, tolerance=0.0001):
        '''
        Secant Method.
        Ref https://en.wikipedia.org/wiki/Secant_method
        '''
        xprev = a
        x = b
        fx = func(x)
        
        for i in range(max_it):

            x_new = x - fx*(x-xprev)/(fx-func(xprev))

            # Swap Variable
            xprev = x
            x = x_new

            fx = func(x)
            error = abs(fx)

            if error <= tolerance:
                return x   
        print('Warining: Reach maximum itteration!')    
        return x  

# --------------------------------------------------------------------------------------------------------
# Groupping Fucntion
# --------------------------------------------------------------------------------------------------------

def group_n_members(member_list, members_per_group):
  return [member_list[i:i + members_per_group] for i in range(0,len(member_list),members_per_group)]

def devide_n_group(member_list, number_groups):
  no_members = len(member_list)
  min_members_per_group = no_members // number_groups
  residual = no_members % number_groups
  if no_members % number_groups == 0:
      min_members_per_group = no_members // number_groups
      return group_n_members(member_list, min_members_per_group)
  else:
      return (group_n_members(member_list[:min_members_per_group*residual + residual], min_members_per_group+1)
      + group_n_members(member_list[min_members_per_group*residual + residual:], min_members_per_group))

# --------------------------------------------------------------------------------------------------------
# Testing 
# --------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Example of numerical integrate
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    y = [1,2,3,4,5,6,7,8,9,10,11,12,13] 

    # Left Riemann technique
    print(NumericalIntegrate().lriemann(x, y))

    # Right Riemann technique
    print(NumericalIntegrate().rriemann(x, y))

    # Trapezoidal technique
    print(NumericalIntegrate().trapezoidal(x, y))

    # Trapezoidal upper lower technique
    print(NumericalIntegrate().trapezoidal_upper_lower(x, y))

    # Simpson13 technique
    print(NumericalIntegrate().simpson13(x, y))

    # Simpson38 technique
    print(NumericalIntegrate().simpson38(x, y))

    # Auto Simpson technique
    print(NumericalIntegrate().autosimpson(x, y))

# -----------------------------------------------------------------------------------------------------------

# Example of numerical root finding
    import timeit
    
    def eq(x):
        return 2 * math.exp(x) + x - 4

    solver = NumericalRootFinding()

    # Method of false position
    print("\nMethod of false position")
    start = timeit.default_timer()
    print("Solution",solver.regula_falsi(eq,0,1, max_it=1000000, tolerance=0.00000001))
    print("Error",eq(solver.regula_falsi(eq,0,1, max_it=1000000, tolerance=0.00000001)))
    end = timeit.default_timer()
    print("time = ", end - start)

    # Bisection Method
    print("\nBisection Method")
    start = timeit.default_timer()
    print("Solution",solver.bisection(eq,0,1, max_it=1000000, tolerance=0.00000001))
    print("Error",eq(solver.bisection(eq,0,1, max_it=1000000, tolerance=0.00000001)))
    end = timeit.default_timer()
    print("time = ", end - start)

    # Secant Method
    print("\nSecant Method")
    start = timeit.default_timer()
    print("Solution",solver.secant(eq,0,1, max_it=1000000, tolerance=0.00000001))
    print("Error",eq(solver.secant(eq,0,1, max_it=1000000, tolerance=0.00000001)))
    end = timeit.default_timer()
    print("time = ", end - start)

# -----------------------------------------------------------------------------------------------------------
