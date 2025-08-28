COLORS = dict(MPL_BLUE_RGB=(31, 119, 180),
              MPL_ORANGE_RGB=(255, 127, 14),
              MPL_GREEN_RGB=(44, 160, 44),
              MPL_RED_RGB=(214, 39, 40),
              OFF_WHITE_RGB=(246, 238, 227),
              DARK_NAVY_RGB=(0, 4, 51),
              DARK_RED_RGB=(179, 25, 66),
              GREEN=(0, 255, 0),
              RED=(255, 0, 0),
              DARK_GRAY=(100, 100, 100),  # background against state value images
              SKY_BLUE=(135, 206, 235),
              NEON_GREEN=(57, 255, 20),
              NEON_BLUE=(31, 81, 255),
              NEON_RED=(255, 20, 47))


MPL_CYCLE_COLORS = [(31, 119, 180),
                    (255, 127, 14),
                    (214, 39, 40),
                    (148, 103, 189),
                    (227, 119, 194),
                    (150, 75, 0),
                    (188, 189, 34),
                    (150, 150, 150),
                    (44, 160, 44),
                    (0, 4, 51)] * 10

# analogy_source : analogy_dest :: analogy_input : analogy_output
COLOR_SCHEME = {'mouseover': COLORS['NEON_GREEN'],
                'a_source': COLORS['DARK_RED_RGB'],
                'a_dest': COLORS['NEON_RED'],
                'a_input': COLORS['DARK_NAVY_RGB'],
                'a_output': COLORS['SKY_BLUE'],
                'vector': COLORS['GREEN'],
                'bkg': COLORS['OFF_WHITE_RGB'],
                'text': COLORS['DARK_NAVY_RGB'], }


def bgr2rgb(color):
    return color[2], color[1], color[0]
