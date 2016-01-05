#!/usr/bin/env python
import cv2
import numpy as np


class Color(object):
    def __init__(self):
        self.colors = list()
        #self.colors.append({'name': 'name', 'R': 0, 'G': 0, 'B': 0})

    def setColorHex(self):
        for cData in self._colors:
            h, n = cData
            r = "0x" + h[0:2]
            g = "0x" + h[2:4]
            b = "0x" + h[4:6]
            self.colors.append({'name': n, 'R': int(r, 0), 'G': int(g, 0), 'B': int(b, 0)})

    def setColorRGB(self):
        for cData in self._colors:
            v, n = cData
            r = v[0]
            g = v[1]
            b = v[2]
            self.colors.append({'name': n, 'R': r, 'G': g, 'B': b})

    def show(self):
        n = len(self.colors)
        w = 250
        h = 20
        img = np.zeros((30 * h if n > 30 else n * h, w * (1 + n / 30), 3), dtype=np.uint8)

        i = 0
        for c in self.colors:
            r = c.get('R')
            g = c.get('G')
            b = c.get('B')

            # set bg color
            x = (i / 30) * w
            y = (i % 30) * h
            img[y:y + h, x:x + w, :] = [b, g, r]

            val = c.get('R') * 256 * 256 + \
                c.get('G') * 256 + \
                c.get('B')

            text = "#" + hex(val)[2:].zfill(6).upper() + \
                " " + c.get('name')

            # set text color
            textColor = None
            if val > int(0xffffff) / 2.:
                textColor = (0, 0, 0)
            else:
                textColor = (255, 255, 255)

            posX = 5 + (i / 30) * w
            posY = int(h * 0.7 + (i % 30) * h)
            cv2.putText(img, text, (posX, posY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        textColor)
            i += 1

        cv2.imshow("Bg1", img)
        while True:
            key = cv2.waitKey(0) & 255
            if key == 27:
                break


# Macbeth ColorChecker (24 colors)
class Macbeth(Color):
    def __init__(self):
        super(Macbeth, self).__init__()
        self._colors = [
            ['735244', 'Dark Skin'],
            ['c29682', 'Light Skin'],
            ['627a9d', 'Blue Sky'],
            ['576c43', 'Foliage'],
            ['8580b1', 'Blue Flower'],
            ['67bdaa', 'Bluish Green'],
            ['d67e2c', 'Orange'],
            ['505ba6', 'Purplish Blue'],
            ['c15a63', 'Moderate Red'],
            ['5e3c6c', 'Purple'],
            ['9dbc40', 'Yellow Green'],
            ['e0a32e', 'Orange Yellow'],
            ['383d96', 'Blue'],
            ['469449', 'Green'],
            ['af363c', 'Red'],
            ['e7c71f', 'Yellow'],
            ['bb5695', 'Magenta'],
            ['0885a1', 'Cyan'],
            ['f3f3f2', 'White'],
            ['c8c8c8', 'Neutral 8'],
            ['a0a0a0', 'Neutral 6.5'],
            ['7a7a79', 'Neutral 5'],
            ['555555', 'Neutral 3.5'],
            ['343434', 'Black'],
        ]

        self.setColorHex()


class MRMRS(Color):
    def __init__(self):
        super(MRMRS, self).__init__()
        self._colors = [
            ['001f3f', 'NAVY'],
            ['0074D9', 'BLUE'],
            ['7FDBFF', 'AQUA'],
            ['39CCCC', 'TEAL'],
            ['3D9970', 'OLIVE'],
            ['2ECC40', 'GREEN'],
            ['01FF70', 'LIME'],
            ['FFDC00', 'YELLOW'],
            ['FF851B', 'ORANGE'],
            ['FF4136', 'RED'],
            ['85144b', 'MAROON'],
            ['F012BE', 'FUCHSIA'],
            ['B10DC9', 'PURPLE'],
            ['111111', 'BLACK'],
            ['AAAAAA', 'GRAY'],
            ['DDDDDD', 'SILVER'],
        ]

        self.setColorHex()


class CSS3(Color):
    def __init__(self):
        super(CSS3, self).__init__()
        self._colors = [
            ['F0F8FF', 'aliceblue'],
            ['2F4F4F', 'darkslategray'],
            ['FFA07A', 'lightsalmon'],
            ['DB7093', 'palevioletred'],
            ['FAEBD7', 'antiquewhite'],
            ['00CED1', 'darkturquoise'],
            ['20B2AA', 'lightseagreen'],
            ['FFEFD5', 'papayawhip'],
            ['00FFFF', 'aqua'],
            ['9400D3', 'darkviolet'],
            ['87CEFA', 'lightskyblue'],
            ['FFEFD5', 'peachpuff'],
            ['7FFFD4', 'aquamarine'],
            ['FF1493', 'deeppink'],
            ['778899', 'lightslategray'],
            ['CD853F', 'peru'],
            ['F0FFFF', 'azure'],
            ['00BFFF', 'deepskyblue'],
            ['B0C4DE', 'lightsteelblue'],
            ['FFC0CB', 'pink'],
            ['F5F5DC', 'beige'],
            ['696969', 'dimgray'],
            ['FFFFE0', 'lightyellow'],
            ['DDA0DD', 'plum'],
            ['FFE4C4', 'bisque'],
            ['1E90FF', 'dodgerblue'],
            ['00FF00', 'lime'],
            ['B0E0E6', 'powderblue'],
            ['000000', 'black'],
            ['B22222', 'firebrick'],
            ['32CD32', 'limegreen'],
            ['800080', 'purple'],
            ['FFFFCD', 'blanchedalmond'],
            ['FFFAF0', 'floralwhite'],
            ['FAF0E6', 'linen'],
            ['FF0000', 'red'],
            ['0000FF', 'blue'],
            ['228B22', 'forestgreen'],
            ['FF00FF', 'magenta'],
            ['BC8F8F', 'rosybrown'],
            ['8A2BE2', 'blueviolet'],
            ['FF00FF', 'fuchsia'],
            ['800000', 'maroon'],
            ['4169E1', 'royalblue'],
            ['A52A2A', 'brown'],
            ['DCDCDC', 'gainsboro'],
            ['66CDAA', 'mediumaquamarine'],
            ['8B4513', 'saddlebrown'],
            ['DEB887', 'burlywood'],
            ['F8F8FF', 'ghostwhite'],
            ['0000CD', 'mediumblue'],
            ['FA8072', 'salmon'],
            ['5F9EA0', 'cadetblue'],
            ['FFD700', 'gold'],
            ['BA55D3', 'mediumorchid'],
            ['F4A460', 'sandybrown'],
            ['7FFF00', 'chartreuse'],
            ['DAA520', 'goldenrod'],
            ['9370DB', 'mediumpurple'],
            ['2E8B57', 'seagreen'],
            ['D2691E', 'chocolate'],
            ['808080', 'gray'],
            ['3CB371', 'mediumseagreen'],
            ['FFF5EE', 'seashell'],
            ['FF7F50', 'coral'],
            ['008000', 'green'],
            ['7B68EE', 'mediumslateblue'],
            ['A0522D', 'sienna'],
            ['6495ED', 'cornflowerblue'],
            ['ADFF2F', 'greenyellow'],
            ['00FA9A', 'mediumspringgreen'],
            ['C0C0C0', 'silver'],
            ['FFF8DC', 'cornsilk'],
            ['F0FFF0', 'honeydew'],
            ['48D1CC', 'mediumturquoise'],
            ['87CEEB', 'skyblue'],
            ['DC143C', 'crimson'],
            ['FF69B4', 'hotpink'],
            ['C71385', 'mediumvioletred'],
            ['6A5ACD', 'slateblue'],
            ['00FFFF', 'cyan'],
            ['CD5C5C', 'indianred'],
            ['191970', 'midnightblue'],
            ['708090', 'slategray'],
            ['00008B', 'darkblue'],
            ['4B0082', 'indigo'],
            ['F5FFFA', 'mintcream'],
            ['FFFAFA', 'snow'],
            ['008B8B', 'darkcyan'],
            ['FFF0F0', 'ivory'],
            ['FFE4E1', 'mistyrose'],
            ['00FF7F', 'springgreen'],
            ['A9A9A9', 'darkgray'],
            ['E6E6FA', 'lavender'],
            ['FFDEAD', 'navajowhite'],
            ['D2B48C', 'tan'],
            ['006400', 'darkgreen'],
            ['FFF0F5', 'lavenderblush'],
            ['000080', 'navy'],
            ['008080', 'teal'],
            ['BDB76B', 'darkkhaki'],
            ['7CFC00', 'lawngreen'],
            ['FDF5E6', 'oldlace'],
            ['D8BFD8', 'thistle'],
            ['8B008B', 'darkmagenta'],
            ['FFFACD', 'lemonchiffon'],
            ['808000', 'olive'],
            ['FF6347', 'tomato'],
            ['556B2F', 'darkolivegreen'],
            ['ADD8E6', 'lightblue'],
            ['6B8E23', 'olivedrab'],
            ['40E0D0', 'turquoise'],
            ['FF8C00', 'darkorange'],
            ['F08080', 'lightcoral'],
            ['FFA500', 'orange'],
            ['EE82EE', 'violet'],
            ['9932CC', 'darkorchid'],
            ['E0FFFF', 'lightcyan'],
            ['FF4500', 'orangered'],
            ['F5DEB3', 'wheat'],
            ['8B0000', 'darkred'],
            ['FAFAD2', 'lightgoldenrodyellow'],
            ['DA70D6', 'orchid'],
            ['FFFFFF', 'white'],
            ['E9967A', 'darksalmon'],
            ['90EE90', 'lightgreen'],
            ['EEE8AA', 'palegoldenrod'],
            ['F5F5F5', 'whitesmoke'],
            ['8FBC8F', 'darkseagreen'],
            ['D3D3D3', 'lightgrey'],
            ['98FB98', 'palegreen'],
            ['FFFF00', 'yellow'],
            ['483D8B', 'darkslateblue'],
            ['FFB6C1', 'lightpink'],
            ['AFEEEE', 'paleturquoise'],
            ['9ACD32', 'yellowgreen'],
        ]

        self.setColorHex()


class X11(Color):
    def __init__(self):
        super(X11, self).__init__()
        self._colors = [
            ['FFC0CB', 'Pink'],
            ['FFB6C1', 'LightPink'],
            ['FF69B4', 'HotPink'],
            ['FF1493', 'DeepPink'],
            ['DB7093', 'PaleVioletRed'],
            ['C71585', 'MediumVioletRed'],
            ['FFA07A', 'LightSalmon'],
            ['FA8072', 'Salmon'],
            ['E9967A', 'DarkSalmon'],
            ['F08080', 'LightCoral'],
            ['CD5C5C', 'IndianRed'],
            ['DC143C', 'Crimson'],
            ['B22222', 'FireBrick'],
            ['8B0000', 'DarkRed'],
            ['FF0000', 'Red'],
            ['FF4500', 'OrangeRed'],
            ['FF6347', 'Tomato'],
            ['FF7F50', 'Coral'],
            ['FF8C00', 'DarkOrange'],
            ['FFA500', 'Orange'],
            ['FFFF00', 'Yellow'],
            ['FFFFE0', 'LightYellow'],
            ['FFFACD', 'LemonChiffon'],
            ['FAFAD2', 'LightGoldenrodYellow'],
            ['FFEFD5', 'PapayaWhip'],
            ['FFE4B5', 'Moccasin'],
            ['FFDAB9', 'PeachPuff'],
            ['EEE8AA', 'PaleGoldenrod'],
            ['F0E68C', 'Khaki'],
            ['BDB76B', 'DarkKhaki'],
            ['FFD700', 'Gold'],
            ['FFF8DC', 'Cornsilk'],
            ['FFEBCD', 'BlanchedAlmond'],
            ['FFE4C4', 'Bisque'],
            ['FFDEAD', 'NavajoWhite'],
            ['F5DEB3', 'Wheat'],
            ['DEB887', 'BurlyWood'],
            ['D2B48C', 'Tan'],
            ['BC8F8F', 'RosyBrown'],
            ['F4A460', 'SandyBrown'],
            ['DAA520', 'Goldenrod'],
            ['B8860B', 'DarkGoldenrod'],
            ['CD853F', 'Peru'],
            ['D2691E', 'Chocolate'],
            ['8B4513', 'SaddleBrown'],
            ['A0522D', 'Sienna'],
            ['A52A2A', 'Brown'],
            ['800000', 'Maroon'],
            ['556B2F', 'DarkOliveGreen'],
            ['808000', 'Olive'],
            ['6B8E23', 'OliveDrab'],
            ['9ACD32', 'YellowGreen'],
            ['32CD32', 'LimeGreen'],
            ['00FF00', 'Lime'],
            ['7CFC00', 'LawnGreen'],
            ['7FFF00', 'Chartreuse'],
            ['ADFF2F', 'GreenYellow'],
            ['00FF7F', 'SpringGreen'],
            ['00FA9A', 'MediumSpringGreen'],
            ['90EE90', 'LightGreen'],
            ['98FB98', 'PaleGreen'],
            ['8FBC8F', 'DarkSeaGreen'],
            ['3CB371', 'MediumSeaGreen'],
            ['2E8B57', 'SeaGreen'],
            ['228B22', 'ForestGreen'],
            ['008000', 'Green'],
            ['006400', 'DarkGreen'],
            ['66CDAA', 'MediumAquamarine'],
            ['00FFFF', 'Aqua'],
            ['00FFFF', 'Cyan'],
            ['E0FFFF', 'LightCyan'],
            ['AFEEEE', 'PaleTurquoise'],
            ['7FFFD4', 'Aquamarine'],
            ['40E0D0', 'Turquoise'],
            ['48D1CC', 'MediumTurquoise'],
            ['00CED1', 'DarkTurquoise'],
            ['20B2AA', 'LightSeaGreen'],
            ['5F9EA0', 'CadetBlue'],
            ['008B8B', 'DarkCyan'],
            ['008080', 'Teal'],
            ['B0C4DE', 'LightSteelBlue'],
            ['B0E0E6', 'PowderBlue'],
            ['ADD8E6', 'LightBlue'],
            ['87CEEB', 'SkyBlue'],
            ['87CEFA', 'LightSkyBlue'],
            ['00BFFF', 'DeepSkyBlue'],
            ['1E90FF', 'DodgerBlue'],
            ['6495ED', 'CornflowerBlue'],
            ['4682B4', 'SteelBlue'],
            ['4169E1', 'RoyalBlue'],
            ['0000FF', 'Blue'],
            ['0000CD', 'MediumBlue'],
            ['00008B', 'DarkBlue'],
            ['000080', 'Navy'],
            ['191970', 'MidnightBlue'],
            ['E6E6FA', 'Lavender'],
            ['D8BFD8', 'Thistle'],
            ['DDA0DD', 'Plum'],
            ['EE82EE', 'Violet'],
            ['DA70D6', 'Orchid'],
            ['FF00FF', 'Fuchsia'],
            ['FF00FF', 'Magenta'],
            ['BA55D3', 'MediumOrchid'],
            ['9370DB', 'MediumPurple'],
            ['8A2BE2', 'BlueViolet'],
            ['9400D3', 'DarkViolet'],
            ['9932CC', 'DarkOrchid'],
            ['8B008B', 'DarkMagenta'],
            ['800080', 'Purple'],
            ['4B0082', 'Indigo'],
            ['483D8B', 'DarkSlateBlue'],
            ['663399', 'RebeccaPurple'],
            ['6A5ACD', 'SlateBlue'],
            ['7B68EE', 'MediumSlateBlue'],
            ['FFFFFF', 'White'],
            ['FFFAFA', 'Snow'],
            ['F0FFF0', 'Honeydew'],
            ['F5FFFA', 'MintCream'],
            ['F0FFFF', 'Azure'],
            ['F0F8FF', 'AliceBlue'],
            ['F8F8FF', 'GhostWhite'],
            ['F5F5F5', 'WhiteSmoke'],
            ['FFF5EE', 'Seashell'],
            ['F5F5DC', 'Beige'],
            ['FDF5E6', 'OldLace'],
            ['FFFAF0', 'FloralWhite'],
            ['FFFFF0', 'Ivory'],
            ['FAEBD7', 'AntiqueWhite'],
            ['FAF0E6', 'Linen'],
            ['FFF0F5', 'LavenderBlush'],
            ['FFE4E1', 'MistyRose'],
            ['DCDCDC', 'Gainsboro'],
            ['D3D3D3', 'LightGrey'],
            ['C0C0C0', 'Silver'],
            ['A9A9A9', 'DarkGray'],
            ['808080', 'Gray'],
            ['696969', 'DimGray'],
            ['778899', 'LightSlateGray'],
            ['708090', 'SlateGray'],
            ['2F4F4F', 'DarkSlateGray'],
            ['000000', 'Black'],
        ]

        self.setColorHex()


class ColorSet01(Color):
    def __init__(self):
        super(ColorSet01, self).__init__()
        self._colors = [
            [(255, 255, 255), 'white'],
            [(0, 0, 0), 'black'],
            [(255, 0, 0), 'red'],
            [(139, 0, 0), 'darkred'],
            [(0, 255, 0), 'green'],
            [(173, 255, 47), 'greenyellow'],
            [(0, 100, 0), 'darkgreen'],
            [(0, 0, 255), 'blue'],
            [(0, 0, 139), 'darkblue'],
            [(0, 191, 255), 'lightblue'],
            [(255, 255, 0), 'yellow'],
            [(0, 255, 255), 'cyan'],
            [(255, 0, 255), 'lightpurple'],
            [(192, 192, 192), 'lightgray'],
            [(255, 165, 0), 'orange'],
            [(255, 69, 0), 'orangered'],
            [(49, 26, 0), 'darkgold'],
            [(40, 0, 0), 'darkbrown'],
        ]

        self.setColorRGB()


if __name__ == "__main__":
    colorList = [Macbeth(), MRMRS(), CSS3(), X11(), ColorSet01()]
    for c in colorList:
        c.show()
