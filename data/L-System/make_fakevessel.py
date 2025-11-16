import os
import turtle
import numpy as np
import random
import argparse
import math
from PIL import Image

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call example: progress_bar(i + 1, num_images, prefix='Progress:', suffix='Complete', length=50)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def draw_background(a_turtle):
    """ Draw a background rectangle. """
    ts = a_turtle.getscreen()
    canvas = ts.getcanvas()
    height = ts.getcanvas()._canvas.winfo_height()
    width = ts.getcanvas()._canvas.winfo_width()
    turtleheading = a_turtle.heading()
    turtlespeed = a_turtle.speed()
    penposn = a_turtle.position()
    penstate = a_turtle.pen()
    a_turtle.penup()
    a_turtle.speed(0)
    a_turtle.goto(-width/2-2, -height/2+3)
    a_turtle.fillcolor(255,255,255)
    a_turtle.begin_fill()
    a_turtle.setheading(0)
    a_turtle.forward(width)
    a_turtle.setheading(90)
    a_turtle.forward(height)
    a_turtle.setheading(180)
    a_turtle.forward(width)
    a_turtle.setheading(270)
    a_turtle.forward(height)
    a_turtle.end_fill()
    a_turtle.penup()
    a_turtle.setposition(*penposn)
    a_turtle.pen(penstate)
    a_turtle.setheading(turtleheading)
    a_turtle.speed(turtlespeed)

def makecolor(turtle,new_length,width,x,y,theta):
    turtle_screen.colormode(255)
    random_R = np.random.randint(0,255)
    random_G = np.random.randint(0, 255)
    random_B = np.random.randint(0, 255)
    R_insity = random_R
    G_insity = random_G
    B_insity = random_B
    Ar1 = np.random.uniform(-3,3)
    Ag1 = np.random.uniform(-3,3)
    Ab1 = np.random.uniform(-3,3)
    turtle.pu()
    turtle.goto((x, y))
    turtle.setheading(theta)
    width = width
    if width<2:
        width = 2
    turtle.pensize(width)
    new_instenisty_r = int(np.clip(R_insity + Ar1,0,255))
    new_instenisty_g = int(np.clip(G_insity + Ag1,0,255))
    new_instenisty_b = int(np.clip(B_insity + Ab1,0,255))
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    turtle.pd()
    turtle.forward(new_length)
    turtle.pu()
    turtle.goto((x, y))
    turtle.setheading(theta)
    turtle.pd()
    turtle.pencolor(new_instenisty_r, new_instenisty_g, new_instenisty_b)
    turtle.forward(new_length)

class LSystem_vessel():
    def __init__(self, axiom, rules, rules_2, rules_3, theta=0, width=5, dtheta_1=40, dtheta_2=30, start=(-350,0), length=80,iteration=3,width_1=0.79, width_2 = 0.5):
        self.sentence = axiom
        self.rules = rules
        self.relus_2 = rules_2
        self.rules_3 = rules_3
        self.iteration = iteration
        self.width = width
        self.theta = theta
        self.dtheta_1 = dtheta_1
        self.dtheta_2 = dtheta_2
        self.length = length
        self.positions = []
        self.start = start
        self.lamda_1 =  width_1
        self.lamda_2 = width_2
        self.x, self.y = start
        self.lines = []

    def __str__(self):
        return self.sentence

    def generate(self):
        self.x, self.y = self.start
        for iter in range(self.iteration):
            newStr = ""
            for char in self.sentence:
                mapped = char
                try:
                    p = random.random()
                    if p>0.4:
                        mapped = self.rules[char]
                    elif 0.8>p>=0.4:
                        mapped = self.rules_3[char]
                    else:
                        mapped = self.relus_2[char]
                except:
                    pass
                newStr += mapped
            self.sentence = newStr

    def draw(self, turtle):
        turtle.pu()
        turtle.hideturtle()
        turtle.speed(0)
        turtle.goto((self.x, self.y))
        turtle.setheading(self.theta)
        flag = False
        for char in self.sentence:
            turtle.pd()
            if char == 'F' or char == 'G':
                turtle.pu()
                turtle.setheading(self.theta)
                if flag == True:
                    new_length = self.length*self.lamda_1
                else:
                    new_length = self.length*self.lamda_2
                turtle.pensize(self.width)
                x, y = turtle.position()
                theta = self.theta
                width = self.width
                makecolor(turtle, new_length, width, x, y, theta)
                self.x, self.y = turtle.position()
                x_end, y_end = turtle.position()
                self.lines.append({
                    "x_start": x, "y_start": y,
                    "x_end": x_end, "y_end": y_end,
                    "angle": theta, "width": width
                })
            elif char == '+':
                dtheta = np.random.randint(low=10, high=40)
                self.theta += dtheta
                self.width = self.width * self.lamda_1
                turtle.right(self.theta)
            elif char == '-':
                self.width = self.width * self.lamda_2
                dtheta = np.random.randint(low=10, high=40)
                self.theta -= dtheta
                turtle.left(self.theta)
            elif char == '[':
                self.positions.append({'x': self.x, 'y': self.y, 'theta': self.theta, 'width': self.width, 'length': self.length,"new_width_0": self.width * 0.79})
                flag = True
            elif char == ']':
                turtle.pu()
                position = self.positions.pop()
                self.x, self.y, self.theta, self.width = position['x'], position['y'], position['theta'], position['width']
                flag = False
                turtle.goto((self.x, self.y))
                turtle.setheading(self.theta)

def save_txt(txt_data_folder, i, x1_top, y1_top, x2_top, y2_top, x1_bottom, y1_bottom, x2_bottom, y2_bottom):
    def is_out_of_range(x, y):
        return not (-450 <= x <= 450 and -450 <= y <= 450)
    if any(is_out_of_range(x, y) for x in [x1_top, x1_bottom, x2_top, x2_bottom] for y in
           [y1_top, y1_bottom, y2_top, y2_bottom]):
        # print("Out of range, Exit Func。")
        return
    # print((f"Original data: {x1_top},{y1_top},{x2_top},{y2_top},{x2_bottom},{y2_bottom},{x1_bottom},{y1_bottom}"))
    def convert_to_512x512(x, y):
        x_new = (x + 389.5) * 512.0 / 779.0
        y_new = (389.5 - y) * 512.0 / 779.0
        return x_new, y_new
    x1_top, y1_top = convert_to_512x512(x1_top, y1_top)
    x1_bottom, y1_bottom = convert_to_512x512(x1_bottom, y1_bottom)
    x2_top, y2_top = convert_to_512x512(x2_top, y2_top)
    x2_bottom, y2_bottom = convert_to_512x512(x2_bottom, y2_bottom)
    if not os.path.exists(txt_data_folder):
        os.makedirs(txt_data_folder)
    file_name = os.path.join(txt_data_folder, f"{i}.txt")
    with open(file_name, 'a') as f:
        f.write(f"{x1_top},{y1_top},{x2_top},{y2_top},{x2_bottom},{y2_bottom},{x1_bottom},{y1_bottom},crack\n")

def draw_tilted_rectangle(x1, y1, x2, y2, width):
    dx = x2 - x1
    dy = y2 - y1
    if 0 < width <= 1.2:
        length_factor = 8
    elif 1.2 < width < 2.2:
        length_factor = 4.8
    elif 2.2 <= width < 3.95:
        length_factor = 2.5
    else:
        length_factor = 1.5
    length = math.sqrt(dx ** 2 + dy ** 2)
    dx = dx * length_factor
    dy = dy * length_factor
    unit_dx = dx / length
    unit_dy = dy / length
    perp_dx = -unit_dy
    perp_dy = unit_dx
    half_width = width * np.random.uniform(2.7, 3)
    x1_top = x1 + perp_dx * half_width
    y1_top = y1 + perp_dy * half_width
    x1_bottom = x1 - perp_dx * half_width
    y1_bottom = y1 - perp_dy * half_width
    x2_top = x2 + perp_dx * half_width
    y2_top = y2 + perp_dy * half_width
    x2_bottom = x2 - perp_dx * half_width
    y2_bottom = y2 - perp_dy * half_width
    save_txt('txt_data', i, x1_top, y1_top, x2_top, y2_top, x1_bottom, y1_bottom, x2_bottom, y2_bottom)

    '''''''''''
    turtle.pencolor(0, 255, 0)  # 设置画笔颜色为绿色
    turtle.pu()

    turtle.goto(x1_top, y1_top)  # 移动到矩形的第一个点
    turtle.pd()
    turtle.goto(x2_top, y2_top)  # 绘制第一条边
    turtle.goto(x2_bottom, y2_bottom)  # 绘制第二条边
    turtle.goto(x1_bottom, y1_bottom)  # 绘制第三条边
    turtle.goto(x1_top, y1_top)  # 绘制第四条边（闭合矩形）
    turtle.pu()
    turtle.pencolor(0, 0, 0)  # 设置画笔颜色为黑色
    '''''''''''

# You can change to finetune the shape of curve structure
# rules = {"F":"F-F+F[+F-F+F-F][-F+F]"}
# rules_2 = {"F":"F+F+F-F[-F+F]"}
rules = {"F":"F-F[+F-F][-F+F]"}
rules_2 = {"F": "F+F[+F[+F]-F]-F+F[+F-F[+F]-F]"}
rules_3 = {"F": "F[+F]+F[+F]+F[+F]"}
rules_4 = {"F": "F-F-F[+F-F][-F-F]F+F"}
path = "[+F-F][-F]"
print(path)
Start_theta = (-80,20)
Start_theta_2 = (170,210)
Start_position_x = (-350, -150)
Start_position_y = (-100, -100)
Start_position_x2 = (150, 350)
Ratio_LW = (0.7,1)
Dtheta = (20,120)
Width = (2, 8)
Length_range = (120, 200)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fake vessel images')
    parser.add_argument('--num', type=int, default=1000, help='Number of images to generate')
    args = parser.parse_args()
    
    num_images = args.num  # Get number from command line argument
    
    print(f"Starting generation of {num_images} vessel images...")
    for i  in range(0, num_images):
        p2 = random.random()
        if p2>0.5:
            path = "[+F-F][-F]"
        else:
            path = "F"
        p_vessel = random.random()
        if p_vessel>0.5:
            r1= rules
            r2= rules_2
            r3 = rules_3
        else:
            r1 = rules_4
            r2 = rules_3
            r3 = rules
        print("image_num", i)
        dtheta_1 = np.random.uniform(Dtheta[0],Dtheta[1])
        dtheta_2 = np.random.uniform(Dtheta[0],Dtheta[1])
        init_width = np.random.randint(Width[0],Width[1]+1)
        init_length = np.random.randint(Length_range[0],Length_range[1]+1)
        iteration = np.random.randint(1,3)

        Ratio_lw_1 = np.random.uniform(Ratio_LW[0],Ratio_LW[1])
        Ratio_lw_2 = np.random.uniform(Ratio_LW[0], Ratio_LW[1])

        p = random.random()
        if p>0.5:
            init_theta = np.random.uniform(Start_theta[0], Start_theta[1])
            x_position = np.random.randint(Start_position_x[0],Start_position_x[1]+1)
        else:
            init_theta = np.random.uniform(Start_theta_2[0], Start_theta_2[1])
            x_position = np.random.randint(Start_position_x2[0],Start_position_x2[1]+1)
        y_position = np.random.randint(Start_position_y[0],Start_position_y[1]+1)

        system = LSystem_vessel(path, r1, r2, r3, theta=init_theta, width=init_width, dtheta_1=dtheta_1, dtheta_2=dtheta_2, start=(x_position, y_position),length=init_length, iteration=iteration, width_1=Ratio_lw_1, width_2=Ratio_lw_2)
        system.generate()
        turtle.speed(0)
        turtle.delay(0)
        turtle.tracer(False)
        turtle_screen = turtle.Screen()  # create graphics window
        turtle_screen.colormode(255)
        turtle.setup(800,800)
        turtle_screen.screensize(800, 800)
        turtle.bgcolor(0,0,0)
        draw_background(turtle)
        system.draw(turtle)

        for line in system.lines:
            x1 = line["x_start"]
            y1 = line["y_start"]
            x2 = line["x_end"]
            y2 = line["y_end"]
            angle = line["angle"]
            width = line["width"]

            draw_tilted_rectangle(x1, y1, x2, y2, width)

        file_name = "./fake_rgbvessel_thin/"+str(i)+'.png'
        output_dir = './fake_rgbvessel_thin'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        tsimg = turtle.getscreen()
        tsimg.getcanvas().postscript(file="work_vessel.eps")
        im = Image.open("work_vessel.eps")
        out = im.resize((512,512))
        im_array = np.array(out)
        print("im_array",im_array.shape)
        out.save(file_name)
        turtle.reset()
        # Show progress bar
        progress_bar(i + 1, num_images, prefix='Generation Progress:', suffix=f'Complete ({i+1}/{num_images})', length=50)
    
    print("Vessel image generation completed!")