
from math import factorial, e, pi
import sympy as smp
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from re import match, sub


class Functionality(BoxLayout):
    def __init__(self, **kwargs):
        super(Functionality, self).__init__(**kwargs)


class BoxLayouts(BoxLayout):
    my_expr = StringProperty()

    def entries(self, widget):
        self.my_expr += str(widget.text)

    def on_factorial(self):
        try:
            if int(self.my_expr) > 366:
                self.my_expr = "Value Too Large!"
                return None
            self.my_expr = str(factorial(int(self.my_expr)))
        except ValueError:
            self.my_expr = "Invalid Input"
            return None

    def on_simplify(self):
        try:
            if self.my_expr == '':
                return None
            self.my_expr = str(smp.simplify(eval(self.my_expr)))
            if "/" in self.my_expr:
                self.my_expr = str(float(self.my_expr))
            if match(r'(\d+)([a-z]+)', self.my_expr):
                self.my_expr = "Math Error!"
        finally:
            return 0

    def constant_pi_e(self):
        while True:
            if "π" in self.my_expr:
                self.my_expr = self.my_expr.replace("π", str(pi))
            if "e" in self.my_expr:
                self.my_expr = self.my_expr.replace("e", str(e))
            break

    def backspacing(self):
        self.my_expr = sub(r'\w$', '', self.my_expr)
        self.my_expr = sub(r'\W$', '', self.my_expr)

    def modify(self, match):
        letter, number = match.groups()
        val = str(number) + '*' + letter
        return val

    def special_functions(self):
        self.my_expr = sub(r'sin', str(smp.sin), self.my_expr)
        self.my_expr = sub(r"∞", str(smp.oo), self.my_expr)
        self.my_expr = sub(r'lim', str(smp.limit), self.my_expr)
        self.my_expr = sub(r'cos', str(smp.cos), self.my_expr)
        self.my_expr = sub(r'tan', str(smp.tan), self.my_expr)
        self.my_expr = sub(r'log', str(smp.log), self.my_expr)
        self.my_expr = sub(r'arcsin', str(smp.asin), self.my_expr)
        self.my_expr = sub(r'arccos', str(smp.acos), self.my_expr)
        self.my_expr = sub(r'arctan', str(smp.atan), self.my_expr)
        self.my_expr = sub(r'sinh', str(smp.sinh), self.my_expr)
        self.my_expr = sub(r'cosh', str(smp.cosh), self.my_expr)
        self.my_expr = sub(r'tanh', str(smp.tanh), self.my_expr)
        self.my_expr = sub(r'atanh', str(smp.atanh), self.my_expr)
        self.my_expr = sub(r'asinh', str(smp.asinh), self.my_expr)
        self.my_expr = sub(r'acosh', str(smp.acosh), self.my_expr)
        self.my_expr = sub(r'integrate', str(smp.integrate), self.my_expr)
        self.my_expr = sub(r'∞', str(smp.zoo), self.my_expr)
        self.my_expr = sub(r'∞', str(00), self.my_expr)

        if int(self.my_expr.count("(")) != int(self.my_expr.count(")")):
            while int(self.my_expr.count("(")) < int(self.my_expr.count(")")):
                self.my_expr += "("
                break
            while int(self.my_expr.count("(")) > int(self.my_expr.count(")")):
                self.my_expr += ")"
                break

    def calculus_integration_with_resp_to_y(self):
        if not self.my_expr:
            return None

        if self.my_expr:
            self.my_expr = sub(r'(\d+)([a-z]+)', self.modify, (self.my_expr))
            self.my_expr = sub(r'([a-zA-Z]+)(\d+)', self.modify_again, (self.my_expr))
            self.special_functions()
            x, y, z = smp.symbols('x y z')
            self.my_expr = str(smp.integrate(self.my_expr, y)) + " + C"
        if "**" in self.my_expr:
            self.my_expr = self.my_expr.replace("**", "^")
        elif "*" in self.my_expr:
            self.my_expr = self.my_expr.replace("*", "")

    def do_lim(self):
        self.special_functions()
        x = smp.symbols('x')
        self.my_expr = str(smp.limit(self.my_expr, x, 00))

    def limits_function(self):
        self.special_functions()
        x = smp.symbols('x')
        if match(r'-$', self.my_expr):
            self.my_expr = str(smp.limit(self.my_expr, x, self.rules, dir="-"))
        else:
            self.my_expr = str(smp.limit(self.my_expr, x, self.rules, dir="+"))

    def calculator_help(self):
        from tkinter import Label, Tk
        root = Tk()
        root.title("IMPORTANT NOTE")

        Label(root, text="In your expressions, try not to omit the multiplication sign(*).\n"
                           "Examples:\n Type 7*x instead of 7x.\n"
                           "Type sin(x)*cos(x) not sin(x)cos(x).\n"
                           "(x)(y) should be typed in as (x)*(y).\n"
                           "Clear the screen before you start a new calculation."
                            , bg="indigo", fg="magenta", font=('sans', 12)).pack()
        root.mainloop()

    def ln_information(self):
        from tkinter import Label, Tk
        mandla = Tk()
        mandla.title("Help Window")
        message = Label(mandla, text="Changing from base 'e'.\n\n\n"
                                     "To change the base of your log,\n\n"
                                     "follow the examples below:\n\n"
                                     "The format: log(your number, base)\n\n"
                                     "example_1: log(100, 10)\n\n"
                                     "example_2: log(29, 2)", font=30, bg="sky blue")
        message.pack()
        mandla.mainloop()

    def modify_again(self, match):
        letter, number = match.groups()
        val = letter + '**' + str(number)
        return val

    def sign_errors(self):
        if match(r'(\+)?=[*/\-]', self.my_expr) or match(r'(\*)?=[-*/+]', self.my_expr) or match(r'(/)?=[+*/-]',
                                                                                                 self.my_expr) or match(
                r'(-)?=[-+*/]', self.my_expr):
            self.my_expr = "Input Error!"
            return 0

    def check_buffer(self):
        if "∞" in self.my_expr:
            self.my_expr = sub(r"∞", "00", self.my_expr)
        if self.my_expr != "":
            while len(self.my_expr) > 0:
                list(self.my_expr).pop()
                break

    def calculus_integration_with_resp_to_x(self):
        if not self.my_expr:
            return None

        if self.my_expr:
            self.my_expr = sub(r'(\d+)([a-z]+)', self.modify, (self.my_expr))
            self.my_expr = sub(r'([a-zA-Z]+)(\d+)', self.modify_again, (self.my_expr))
            self.special_functions()
            x, y, z = smp.symbols('x y z')
            self.my_expr = str(smp.integrate(self.my_expr, x)) + " + C"
        if "**" in self.my_expr:
            self.my_expr = self.my_expr.replace("**", "^")
        elif "*" in self.my_expr:
            self.my_expr = self.my_expr.replace("*", "")
        return 0

    def differential_calculus_x(self):
        if self.my_expr:
            self.special_functions()
            x, y, z = smp.symbols('x y z')
            self.my_expr = str(smp.diff(self.my_expr, x))
        return None

    def differential_calculus_y(self):
        if self.my_expr:
            self.special_functions()
            x, y, z = smp.symbols('x y z')
            self.my_expr = str(smp.diff(self.my_expr, y))
        return 0

    def equal_sign(self):
        self.constant_pi_e()
        self.check_buffer()
        self.sign_errors()
        # self.ids.text = str(0)
        x, y, z = smp.symbols('x y z')
        self.special_functions()

        if self.my_expr == '':
            self.my_expr = str(0)
        else:
            self.my_expr = self.my_expr

        self.my_expr = sub(r'(\d+)([a-z]+)', self.modify, (self.my_expr))
        self.my_expr = sub(r'([a-zA-Z]+)(\d+)', self.modify_again, (self.my_expr))
        self.my_expr = sub(r'->', ',', self.my_expr)

        if match(r'0/0', self.my_expr):
            self.my_expr = "Undefined Value"
            return 0

        elif match(r'(\d+)([a-z]+)', self.my_expr):
            self.my_expr = "Math Error!"
            return 0

        elif match(r'^[*/+-]', self.my_expr):
            self.my_expr = str(0)
            return 0

        expression = str(self.my_expr)
        try:
            self.my_expr = str(smp.simplify(expression))
        except TypeError:
            self.my_expr = "MATH ERROR!"
        except SyntaxError:
            self.my_expr = "MATH ERROR!"
            return None

        if "**" in self.my_expr:
            self.my_expr = self.my_expr.replace("**", "^")
        elif "*" in self.my_expr:
            self.my_expr = self.my_expr.replace("*", "")

    def on_close(self):
        quit()

    def on_clear(self):
        self.my_expr = ''


class MathmapaApp(App):
    def build(self):
        self.title = "Algebraic Terms"


if __name__ == "__main__":
    MathmapaApp().run()
