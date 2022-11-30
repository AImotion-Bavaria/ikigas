from PIL import Image, ImageDraw, ImageFont

def get_img(linepack, valve_position):
    width, height = 500, 300
    color_bg = (128, 128, 128)
    im = Image.new('RGB', (width, height), color_bg)
    draw = ImageDraw.Draw(im)

    white = (255, 255, 255)
    black = (0, 0, 0)
    blue = (4,89,154)
    tub_width = 270
    tub_height = 200

    center_x, center_y = width/2, height/2

    tub_left_x, tub_left_y = (center_x-tub_width/2, center_y-tub_height/2)
    tub_right_x, tub_right_y = (center_x+tub_width/2, center_y+tub_height/2)
    draw.rectangle( ((tub_left_x, tub_left_y), (tub_right_x, tub_right_y)), fill=white, outline=white)

    def draw_water(draw, fill_status=0.5):
        y_start = tub_right_y - (tub_right_y - tub_left_y)*fill_status
        draw.rectangle(((tub_left_x, y_start), (tub_right_x, tub_right_y)), fill=blue, outline=white)

    draw_water(draw, linepack)
    # use a truetype font
    font = ImageFont.truetype("arial.ttf", 16)

    draw.text((20,15), "Valve", font=font)

    # draw the valve - two rectangles essentially
    valve_width = 50
    valve_height = 20
    valve_left_x = tub_left_x/2 - valve_width/2
    valve_left_y = tub_left_y

    draw.rectangle( ((valve_left_x, valve_left_y), (valve_left_x + valve_width, valve_left_y+valve_height)), fill=white, outline=white)

    draw.text((valve_left_x, valve_left_y+valve_height+3), "0", font=font)
    draw.text((valve_left_x + valve_width-3, valve_left_y+valve_height+3), "1", font=font)
    # valve position

    def draw_knob(draw, pos=0.5):
        knob_width = 6
        knob_excess = 3
        knob_height = valve_height + 2*knob_excess
        knob_left_x = valve_left_x + pos*valve_width - knob_width/2
        knob_left_y = valve_left_y - knob_excess
        draw.rectangle(((knob_left_x, knob_left_y), (knob_left_x + knob_width, knob_left_y + knob_height)),
                       fill=black, outline=black)

    draw_knob(draw, valve_position)
    return im