# Beta-3D-Editor-in-python
I do this for every maker.You can change this codes.But if you are interested in commercializing this project or bringing it to market, please contact me.Last I worked on this like 2 days please use it and change it.

The code:
"""
3D Creator — Blender-style desktop app
Pure Python + Tkinter, zero external dependencies.
Run:  python3 creator3d.py
"""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import math, copy, time, json, random, os, threading
import urllib.request, urllib.error

# ─── MATH HELPERS ────────────────────────────────────────────────────────────

def vec3(x=0.0, y=0.0, z=0.0):
    return [float(x), float(y), float(z)]

def vadd(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def vsub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def vscale(a, s):
    return [a[0]*s, a[1]*s, a[2]*s]

def vdot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vcross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def vnorm(a):
    l = math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return [a[0]/l, a[1]/l, a[2]/l] if l > 1e-9 else [0.0, 0.0, 1.0]

def vlen(a):
    return math.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def mat4_identity():
    return [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

def mat4_mul(A, B):
    C = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            for k in range(4):
                C[i][j] += A[i][k]*B[k][j]
    return C

def mat4_apply(M, v):
    x,y,z = v
    w = M[3][0]*x + M[3][1]*y + M[3][2]*z + M[3][3]
    rx = (M[0][0]*x + M[0][1]*y + M[0][2]*z + M[0][3]) / w
    ry = (M[1][0]*x + M[1][1]*y + M[1][2]*z + M[1][3]) / w
    rz = (M[2][0]*x + M[2][1]*y + M[2][2]*z + M[2][3]) / w
    return [rx, ry, rz]

def mat4_translate(tx, ty, tz):
    M = mat4_identity()
    M[0][3]=tx; M[1][3]=ty; M[2][3]=tz
    return M

def mat4_scale(sx, sy, sz):
    M = mat4_identity()
    M[0][0]=sx; M[1][1]=sy; M[2][2]=sz
    return M

def mat4_rotx(a):
    c,s = math.cos(a), math.sin(a)
    M = mat4_identity()
    M[1][1]=c; M[1][2]=-s; M[2][1]=s; M[2][2]=c
    return M

def mat4_roty(a):
    c,s = math.cos(a), math.sin(a)
    M = mat4_identity()
    M[0][0]=c; M[0][2]=s; M[2][0]=-s; M[2][2]=c
    return M

def mat4_rotz(a):
    c,s = math.cos(a), math.sin(a)
    M = mat4_identity()
    M[0][0]=c; M[0][1]=-s; M[1][0]=s; M[1][1]=c
    return M

def object_matrix(obj):
    px,py,pz = obj['position']
    rx,ry,rz = obj['rotation']
    sx,sy,sz = obj['scale']
    M = mat4_mul(mat4_translate(px,py,pz),
         mat4_mul(mat4_roty(ry),
         mat4_mul(mat4_rotx(rx),
         mat4_mul(mat4_rotz(rz),
                  mat4_scale(sx,sy,sz)))))
    return M

# ─── PROJECTION ──────────────────────────────────────────────────────────────

class Camera:
    def __init__(self):
        self.target   = [0.0, 0.0, 0.0]
        self.distance = 10.0
        self.yaw      = math.radians(45)
        self.pitch    = math.radians(25)
        self.fov      = 60.0
        self.width    = 800
        self.height   = 600
        self.near     = 0.1
        self.far      = 1000.0

    def position(self):
        cy = math.cos(self.pitch)
        return [
            self.target[0] + self.distance * cy * math.sin(self.yaw),
            self.target[1] + self.distance * math.sin(self.pitch),
            self.target[2] + self.distance * cy * math.cos(self.yaw),
        ]

    def view_matrix(self):
        eye = self.position()
        f = vnorm(vsub(self.target, eye))
        r = vnorm(vcross(f, [0,1,0]))
        u = vcross(r, f)
        M = mat4_identity()
        M[0] = [ r[0],  r[1],  r[2], -vdot(r, eye)]
        M[1] = [ u[0],  u[1],  u[2], -vdot(u, eye)]
        M[2] = [-f[0], -f[1], -f[2],  vdot(f, eye)]
        M[3] = [0, 0, 0, 1]
        return M

    def proj_matrix(self):
        aspect = self.width / max(self.height, 1)
        f = 1.0 / math.tan(math.radians(self.fov) / 2)
        n, fa = self.near, self.far
        M = [[0]*4 for _ in range(4)]
        M[0][0] = f / aspect
        M[1][1] = f
        M[2][2] = (fa + n) / (n - fa)
        M[2][3] = (2 * fa * n) / (n - fa)
        M[3][2] = -1
        return M

    def project(self, world_pt):
        V = self.view_matrix()
        P = self.proj_matrix()
        vp = mat4_apply(V, world_pt)
        # clip check
        if vp[2] >= -self.near:
            return None
        pp = mat4_apply(P, vp)
        sx = (pp[0] + 1) * 0.5 * self.width
        sy = (1 - (pp[1] + 1) * 0.5) * self.height
        return (sx, sy, -vp[2])

# ─── GEOMETRY GENERATORS ─────────────────────────────────────────────────────

def make_cube():
    v = [
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    faces = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(2,6,7,3),(0,3,7,4),(1,5,6,2)]
    return v, edges, faces

def make_plane():
    v = [[-1,0,-1],[1,0,-1],[1,0,1],[-1,0,1]]
    edges = [(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]
    faces = [(0,1,2,3)]
    return v, edges, faces

def make_sphere(lat=8, lon=12):
    verts = []
    for i in range(lat+1):
        phi = math.pi * i / lat
        for j in range(lon):
            theta = 2*math.pi * j / lon
            verts.append([math.sin(phi)*math.cos(theta), math.cos(phi), math.sin(phi)*math.sin(theta)])
    edges = []
    faces = []
    for i in range(lat):
        for j in range(lon):
            a = i*lon + j
            b = i*lon + (j+1)%lon
            c = (i+1)*lon + (j+1)%lon
            d = (i+1)*lon + j
            edges += [(a,b),(a,d)]
            faces.append((a,b,c,d))
    return verts, edges, faces

def make_icosphere():
    t = (1 + math.sqrt(5)) / 2
    verts = [vnorm(v) for v in [
        [-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],
        [0,-1,t],[0,1,t],[0,-1,-t],[0,1,-t],
        [t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1],
    ]]
    faces_idx = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    edges = set()
    for f in faces_idx:
        edges.add((min(f[0],f[1]), max(f[0],f[1])))
        edges.add((min(f[1],f[2]), max(f[1],f[2])))
        edges.add((min(f[0],f[2]), max(f[0],f[2])))
    return verts, list(edges), [list(f) for f in faces_idx]

def make_cylinder(segs=16):
    verts = []
    for i in range(segs):
        a = 2*math.pi*i/segs
        verts.append([math.cos(a), 1, math.sin(a)])
    for i in range(segs):
        a = 2*math.pi*i/segs
        verts.append([math.cos(a), -1, math.sin(a)])
    verts += [[0,1,0],[0,-1,0]]
    top, bot = 2*segs, 2*segs+1
    edges = []
    faces = []
    for i in range(segs):
        n = (i+1)%segs
        edges += [(i, n),(segs+i, segs+n),(i, segs+i)]
        faces.append((i, n, segs+n, segs+i))
        faces.append((top, i, n))
        faces.append((bot, segs+n, segs+i))
    return verts, edges, faces

def make_cone(segs=16):
    verts = []
    for i in range(segs):
        a = 2*math.pi*i/segs
        verts.append([math.cos(a), -1, math.sin(a)])
    verts += [[0,1,0],[0,-1,0]]
    apex, bot = segs, segs+1
    edges = []
    faces = []
    for i in range(segs):
        n = (i+1)%segs
        edges += [(i,n),(i,apex)]
        faces.append((apex, i, n))
        faces.append((bot, n, i))
    return verts, edges, faces

def make_torus(R=1.0, r=0.35, major=24, minor=12):
    verts = []
    for i in range(major):
        a = 2*math.pi*i/major
        for j in range(minor):
            b = 2*math.pi*j/minor
            x = (R + r*math.cos(b))*math.cos(a)
            y = r*math.sin(b)
            z = (R + r*math.cos(b))*math.sin(a)
            verts.append([x,y,z])
    edges = []
    faces = []
    for i in range(major):
        for j in range(minor):
            a = i*minor+j
            b = i*minor+(j+1)%minor
            c = ((i+1)%major)*minor+(j+1)%minor
            d = ((i+1)%major)*minor+j
            edges += [(a,b),(a,d)]
            faces.append((a,b,c,d))
    return verts, edges, faces

MESH_GEN = {
    'cube':     make_cube,
    'plane':    make_plane,
    'sphere':   lambda: make_sphere(8,12),
    'icosphere':make_icosphere,
    'cylinder': lambda: make_cylinder(16),
    'cone':     lambda: make_cone(16),
    'torus':    make_torus,
}

# ─── SCENE OBJECT ────────────────────────────────────────────────────────────

_uid = 0
def new_uid():
    global _uid; _uid += 1; return _uid

def make_object(shape, name=None):
    gen = MESH_GEN.get(shape)
    if gen is None:
        gen = make_cube
    verts, edges, faces = gen()
    obj = {
        'id':       new_uid(),
        'name':     name or shape.capitalize(),
        'shape':    shape,
        'verts':    verts,
        'edges':    edges,
        'faces':    faces,
        'position': [0.0, 0.0, 0.0],
        'rotation': [0.0, 0.0, 0.0],
        'scale':    [1.0, 1.0, 1.0],
        'color':    random_color(),
        'visible':  True,
        'wireframe':False,
    }
    return obj

def random_color():
    h = random.random()
    s, v = 0.55, 0.75
    r,g,b = hsv_to_rgb(h,s,v)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def hsv_to_rgb(h,s,v):
    if s == 0: return v,v,v
    i = int(h*6); f = h*6-i; p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i%6]

def hex_to_rgb_tuple(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

def darken(color, factor=0.55):
    r,g,b = hex_to_rgb_tuple(color)
    return '#{:02x}{:02x}{:02x}'.format(int(r*factor),int(g*factor),int(b*factor))

def shade(color, normal, light_dir=[0.57,0.77,0.3]):
    n = vnorm(normal)
    d = max(0, vdot(n, vnorm(light_dir)))
    ambient = 0.25
    intensity = ambient + (1-ambient)*d
    r,g,b = hex_to_rgb_tuple(color)
    return '#{:02x}{:02x}{:02x}'.format(
        min(255,int(r*intensity)), min(255,int(g*intensity)), min(255,int(b*intensity)))

def shade_full(color, normal, light_dir, intensity=1.0, ambient=0.25, light_rgb=(255,255,255)):
    n = vnorm(normal)
    d = max(0.0, vdot(n, vnorm(light_dir)))
    r,g,b = hex_to_rgb_tuple(color)
    lr,lg,lb = light_rgb
    factor = ambient + (1.0 - ambient) * d * intensity
    out_r = min(255, int(r * factor * lr / 255))
    out_g = min(255, int(g * factor * lg / 255))
    out_b = min(255, int(b * factor * lb / 255))
    return '#{:02x}{:02x}{:02x}'.format(out_r, out_g, out_b)

def face_normal(verts):
    if len(verts) < 3:
        return [0,1,0]
    a = vsub(verts[1], verts[0])
    b = vsub(verts[2], verts[0])
    return vcross(a, b)

# ─── UNDO HISTORY ────────────────────────────────────────────────────────────

class History:
    def __init__(self, app):
        self.app = app
        self.stack = []
        self.index = -1

    def snapshot(self):
        state = json.dumps([self._serialize(o) for o in self.app.objects])
        if self.index >= 0 and self.stack[self.index] == state:
            return
        self.stack = self.stack[:self.index+1]
        self.stack.append(state)
        self.index = len(self.stack)-1

    def undo(self):
        if self.index > 0:
            self.index -= 1
            self._restore()

    def redo(self):
        if self.index < len(self.stack)-1:
            self.index += 1
            self._restore()

    def _restore(self):
        data = json.loads(self.stack[self.index])
        self.app.objects = [self._deserialize(d) for d in data]
        sel_id = self.app.selected_id
        self.app.selected_id = None
        for o in self.app.objects:
            if o['id'] == sel_id:
                self.app.selected_id = sel_id
                break
        self.app.update_outliner()
        self.app.update_props()
        self.app.redraw()

    def _serialize(self, o):
        return {k: copy.deepcopy(v) for k,v in o.items() if k not in ('verts','edges','faces')}

    def _deserialize(self, d):
        gen = MESH_GEN.get(d['shape'], make_cube)
        verts, edges, faces = gen()
        obj = {**d, 'verts': verts, 'edges': edges, 'faces': faces}
        return obj

# ─── MAIN APPLICATION ─────────────────────────────────────────────────────────

class App:
    # ── init ──────────────────────────────────────────────────────────────────
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Creator")
        self.root.configure(bg='#1a1a1a')
        self.root.geometry('1280x800')

        self.camera     = Camera()
        self.objects    = []
        self.selected_id= None
        self.tool       = 'select'  # select | move | rotate | scale
        self.view_mode  = 'solid'   # solid | wireframe
        self.show_grid  = True
        self.snap       = False
        self.snap_val   = 0.25

        self.dragging   = False
        self.drag_start = None
        self.drag_mode  = None      # 'orbit' | 'pan' | 'transform'
        self.trans_axis = 'XYZ'

        self.fps_label  = None
        self._last_time = time.time()
        self._frames    = 0

        # Lighting state
        self.light_azimuth   = tk.DoubleVar(value=45.0)   # degrees
        self.light_elevation = tk.DoubleVar(value=50.0)   # degrees
        self.light_intensity = tk.DoubleVar(value=1.0)
        self.light_ambient   = tk.DoubleVar(value=0.25)
        self.light_color     = '#ffffff'

        self._build_ui()
        self.history = History(self)

        # default scene
        obj = make_object('cube', 'Cube.001')
        self.objects.append(obj)
        obj2 = make_object('plane', 'Plane.001')
        obj2['position'] = [0.0, -1.0, 0.0]
        obj2['scale']    = [4.0, 4.0, 4.0]
        obj2['color']    = '#3a3a3a'
        self.objects.append(obj2)

        self.selected_id = obj['id']
        self.history.snapshot()

        self.update_outliner()
        self.update_props()
        self.redraw()
        self._schedule_fps()

    # ── UI LAYOUT ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ─ TOP BAR
        tb = tk.Frame(root, bg='#2a2a2a', height=38)
        tb.pack(fill='x', side='top')
        tb.pack_propagate(False)

        tk.Label(tb, text=' ◆ 3D Creator ', bg='#2a2a2a', fg='#e87d0d',
                 font=('Segoe UI', 11, 'bold')).pack(side='left', padx=4)
        self._vline(tb)

        # Tool buttons
        self.tool_btns = {}
        tools = [('Q','select','▶ Select'),('G','move','✛ Move'),
                 ('R','rotate','↻ Rotate'),('S','scale','↔ Scale')]
        for key, t, lbl in tools:
            b = tk.Button(tb, text=lbl, bg='#2a2a2a', fg='#aaa',
                          activebackground='#3a3a3a', activeforeground='#fff',
                          relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                          font=('Segoe UI', 9),
                          command=lambda t=t: self.set_tool(t))
            b.pack(side='left')
            self.tool_btns[t] = b
            self._tooltip(b, f'{lbl}  [{key}]')
        self._vline(tb)

        # View buttons
        views = [('Persp','persp'),('Top','top'),('Front','front'),('Right','right')]
        self.view_btns = {}
        for lbl, v in views:
            b = tk.Button(tb, text=lbl, bg='#2a2a2a', fg='#aaa',
                          activebackground='#3a3a3a', activeforeground='#fff',
                          relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                          font=('Segoe UI', 9),
                          command=lambda v=v: self.set_view(v))
            b.pack(side='left')
            self.view_btns[v] = b
        self._vline(tb)

        tk.Button(tb, text='⬡ Solid', bg='#2a2a2a', fg='#aaa',
                  activebackground='#3a3a3a', activeforeground='#fff',
                  relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  font=('Segoe UI', 9),
                  command=self.toggle_wireframe).pack(side='left')
        self.snap_btn = tk.Button(tb, text='⊕ Snap', bg='#2a2a2a', fg='#666',
                  activebackground='#3a3a3a', activeforeground='#fff',
                  relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  font=('Segoe UI', 9),
                  command=self.toggle_snap)
        self.snap_btn.pack(side='left')
        self._vline(tb)

        tk.Button(tb, text='⬡ Grid', bg='#2a2a2a', fg='#aaa',
                  activebackground='#3a3a3a', activeforeground='#fff',
                  relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  font=('Segoe UI', 9),
                  command=self.toggle_grid).pack(side='left')

        self._vline(tb)
        tk.Button(tb, text='⟳ Undo', bg='#2a2a2a', fg='#aaa',
                  relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  font=('Segoe UI', 9),
                  command=self.undo).pack(side='left')
        tk.Button(tb, text='⟳ Redo', bg='#2a2a2a', fg='#aaa',
                  relief='flat', bd=0, padx=8, pady=6, cursor='hand2',
                  font=('Segoe UI', 9),
                  command=self.redo).pack(side='left')

        self.fps_label = tk.Label(tb, text='FPS: --', bg='#2a2a2a', fg='#555',
                                  font=('Consolas', 9))
        self.fps_label.pack(side='right', padx=10)
        self._vline(tb)
        tk.Button(tb, text='🦙 Ask AI', bg='#2a4a6a', fg='#8bcfff',
                  relief='flat', bd=0, padx=10, pady=6, cursor='hand2',
                  font=('Segoe UI', 9, 'bold'),
                  command=self.open_ai_panel).pack(side='right', padx=4)
        self._vline(tb)
        tk.Button(tb, text='⬆ Export', bg='#e87d0d', fg='white',
                  relief='flat', bd=0, padx=10, pady=6, cursor='hand2',
                  font=('Segoe UI', 9, 'bold'),
                  command=self.export_scene).pack(side='right', padx=4)
        tk.Button(tb, text='⬇ Import', bg='#335533', fg='#8eff8e',
                  relief='flat', bd=0, padx=10, pady=6, cursor='hand2',
                  font=('Segoe UI', 9, 'bold'),
                  command=self.import_scene).pack(side='right', padx=(0,4))

        # ─ MAIN
        main = tk.Frame(root, bg='#1a1a1a')
        main.pack(fill='both', expand=True)

        # ─ LEFT PANEL
        lp = tk.Frame(main, bg='#252525', width=170)
        lp.pack(side='left', fill='y')
        lp.pack_propagate(False)
        self._panel_header(lp, 'Add Object')
        shapes = [('cube','■ Cube'),('sphere','● Sphere'),('icosphere','▲ Icosphere'),
                  ('cylinder','⬛ Cylinder'),('cone','▲ Cone'),('torus','◎ Torus'),
                  ('plane','▬ Plane')]
        for shape, lbl in shapes:
            tk.Button(lp, text=lbl, bg='#252525', fg='#bbb', relief='flat', bd=0,
                      padx=10, pady=5, anchor='w', cursor='hand2',
                      activebackground='#333', activeforeground='#fff',
                      font=('Segoe UI', 9), width=16,
                      command=lambda s=shape: self.add_object(s)).pack(fill='x', padx=4, pady=1)

        # ─ LIGHTING PANEL
        tk.Frame(lp, bg='#111', height=1).pack(fill='x', pady=(6,0))
        self._panel_header(lp, 'Lighting')

        def _slider_row(parent, label, var, from_, to, resolution=0.01, fmt='{:.2f}'):
            row = tk.Frame(parent, bg='#252525')
            row.pack(fill='x', padx=6, pady=2)
            tk.Label(row, text=label, bg='#252525', fg='#888',
                     font=('Segoe UI', 8), width=10, anchor='w').pack(side='left')
            val_lbl = tk.Label(row, text=fmt.format(var.get()),
                               bg='#252525', fg='#ccc', font=('Consolas', 8), width=5)
            val_lbl.pack(side='right')
            def on_slide(v, lbl=val_lbl, fmt=fmt):
                lbl.config(text=fmt.format(float(v)))
                self.redraw()
            sl = tk.Scale(row, variable=var, from_=from_, to=to,
                          orient='horizontal', resolution=resolution,
                          bg='#252525', fg='#aaa', troughcolor='#1e1e1e',
                          activebackground='#e87d0d', highlightthickness=0,
                          sliderlength=12, showvalue=False, bd=0,
                          command=on_slide)
            sl.pack(side='left', fill='x', expand=True)

        _slider_row(lp, 'Azimuth',   self.light_azimuth,   0,   360, 1,    '{:.0f}°')
        _slider_row(lp, 'Elevation', self.light_elevation,  5,    85, 1,    '{:.0f}°')
        _slider_row(lp, 'Intensity', self.light_intensity, 0.0,  3.0, 0.05, '{:.2f}')
        _slider_row(lp, 'Ambient',   self.light_ambient,   0.0,  1.0, 0.01, '{:.2f}')

        # Light color swatch
        lc_row = tk.Frame(lp, bg='#252525')
        lc_row.pack(fill='x', padx=6, pady=4)
        tk.Label(lc_row, text='Color', bg='#252525', fg='#888',
                 font=('Segoe UI', 8), width=10, anchor='w').pack(side='left')
        self._light_swatch = tk.Frame(lc_row, bg=self.light_color,
                                      width=60, height=16, cursor='hand2')
        self._light_swatch.pack(side='left')
        def pick_light_color():
            result = colorchooser.askcolor(color=self.light_color, parent=self.root,
                                           title='Light Color')
            if result[1]:
                self.light_color = result[1]
                self._light_swatch.configure(bg=self.light_color)
                self.redraw()
        self._light_swatch.bind('<Button-1>', lambda e: pick_light_color())

        # Reset button
        def reset_lighting():
            self.light_azimuth.set(45.0)
            self.light_elevation.set(50.0)
            self.light_intensity.set(1.0)
            self.light_ambient.set(0.25)
            self.light_color = '#ffffff'
            self._light_swatch.configure(bg=self.light_color)
            self.redraw()
        tk.Button(lp, text='Reset Lighting', bg='#333', fg='#aaa',
                  relief='flat', bd=0, padx=6, pady=4, cursor='hand2',
                  font=('Segoe UI', 8),
                  command=reset_lighting).pack(fill='x', padx=6, pady=(2,4))

        # ─ VIEWPORT
        vp_frame = tk.Frame(main, bg='#1a1a1a')
        vp_frame.pack(side='left', fill='both', expand=True)

        self.canvas = tk.Canvas(vp_frame, bg='#1c1c1c', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        self._bind_viewport()

        # Status bar
        sb = tk.Frame(vp_frame, bg='#1e1e1e', height=22)
        sb.pack(fill='x', side='bottom')
        sb.pack_propagate(False)
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(sb, textvariable=self.status_var, bg='#1e1e1e', fg='#666',
                 font=('Consolas', 9)).pack(side='left', padx=8)
        self.mode_var = tk.StringVar(value='Object Mode')
        tk.Label(sb, textvariable=self.mode_var, bg='#1e1e1e', fg='#888',
                 font=('Consolas', 9)).pack(side='right', padx=8)

        # ─ RIGHT PANEL
        rp = tk.Frame(main, bg='#252525', width=220)
        rp.pack(side='right', fill='y')
        rp.pack_propagate(False)

        self._panel_header(rp, 'Scene Outliner')
        ol_frame = tk.Frame(rp, bg='#252525', height=160)
        ol_frame.pack(fill='x')
        ol_frame.pack_propagate(False)
        self.outliner_list = tk.Listbox(ol_frame, bg='#1e1e1e', fg='#bbb',
                                         selectbackground='#e87d0d',
                                         selectforeground='white',
                                         relief='flat', bd=0, highlightthickness=0,
                                         font=('Segoe UI', 9), activestyle='none')
        self.outliner_list.pack(fill='both', expand=True, padx=2, pady=2)
        self.outliner_list.bind('<<ListboxSelect>>', self._on_outliner_select)

        tk.Frame(rp, bg='#111', height=1).pack(fill='x')

        self._panel_header(rp, 'Properties')
        props_outer = tk.Frame(rp, bg='#252525')
        props_outer.pack(fill='both', expand=True)
        self.props_canvas = tk.Canvas(props_outer, bg='#252525', highlightthickness=0)
        props_scroll = tk.Scrollbar(props_outer, orient='vertical', command=self.props_canvas.yview)
        self.props_canvas.configure(yscrollcommand=props_scroll.set)
        props_scroll.pack(side='right', fill='y')
        self.props_canvas.pack(side='left', fill='both', expand=True)
        self.props_frame = tk.Frame(self.props_canvas, bg='#252525')
        self._props_win_id = self.props_canvas.create_window((0,0), window=self.props_frame, anchor='nw')
        self.props_frame.bind('<Configure>', lambda e: (
            self.props_canvas.configure(scrollregion=self.props_canvas.bbox('all')),
            self.props_canvas.itemconfig(self._props_win_id, width=e.width)
        ))
        self.props_canvas.bind('<Configure>', lambda e:
            self.props_canvas.itemconfig(self._props_win_id, width=e.width))

        self._highlight_tool()
        self.set_view('persp')

    # ── HELPERS ───────────────────────────────────────────────────────────────
    def _vline(self, parent):
        tk.Frame(parent, bg='#444', width=1).pack(side='left', fill='y', padx=4, pady=6)

    def _panel_header(self, parent, text):
        tk.Label(parent, text=text, bg='#2a2a2a', fg='#aaa',
                 font=('Segoe UI', 8, 'bold'), anchor='w',
                 pady=5, padx=8).pack(fill='x')
        tk.Frame(parent, bg='#111', height=1).pack(fill='x')

    def _tooltip(self, widget, text):
        def show(e):
            self.status_var.set(text)
        def hide(e):
            self.status_var.set('Ready')
        widget.bind('<Enter>', show)
        widget.bind('<Leave>', hide)

    # ── VIEWPORT BINDINGS ─────────────────────────────────────────────────────
    def _bind_viewport(self):
        c = self.canvas
        c.bind('<ButtonPress-1>',    self._vp_press_left)
        c.bind('<B1-Motion>',        self._vp_drag_left)
        c.bind('<ButtonRelease-1>',  self._vp_release)
        c.bind('<ButtonPress-2>',    self._vp_press_mid)
        c.bind('<B2-Motion>',        self._vp_drag_mid)
        c.bind('<ButtonPress-3>',    self._vp_press_right)
        c.bind('<B3-Motion>',        self._vp_drag_right)
        c.bind('<MouseWheel>',       self._vp_scroll)
        c.bind('<Button-4>',         lambda e: self._zoom(-1))
        c.bind('<Button-5>',         lambda e: self._zoom(1))
        c.bind('<Configure>',        self._on_resize)
        self.root.bind('<KeyPress>', self._on_key)
        self.root.focus_set()

    def _vp_press_left(self, e):
        self.drag_start = (e.x, e.y)
        self.dragging = False

    def _vp_drag_left(self, e):
        if self.drag_start is None: return
        dx = e.x - self.drag_start[0]
        dy = e.y - self.drag_start[1]
        if not self.dragging and (abs(dx)+abs(dy)) > 3:
            self.dragging = True
        if not self.dragging: return

        if self.tool != 'select' and self.selected_id is not None:
            self._transform_drag(dx, dy, e)
        else:
            self._orbit_drag(dx, dy)
        self.drag_start = (e.x, e.y)

    def _vp_release(self, e):
        if not self.dragging:
            self._pick(e.x, e.y)
        else:
            if self.tool != 'select' and self.selected_id is not None:
                self.history.snapshot()
        self.dragging = False
        self.drag_start = None
        self.drag_mode = None

    def _vp_press_mid(self, e):
        self.drag_start = (e.x, e.y)
        self.drag_mode = 'pan' if (e.state & 0x0001) else 'orbit'

    def _vp_drag_mid(self, e):
        if self.drag_start is None: return
        dx = e.x - self.drag_start[0]
        dy = e.y - self.drag_start[1]
        if self.drag_mode == 'pan':
            self._pan_drag(dx, dy)
        else:
            self._orbit_drag(dx, dy)
        self.drag_start = (e.x, e.y)

    def _vp_press_right(self, e):
        self.drag_start = (e.x, e.y)

    def _vp_drag_right(self, e):
        if self.drag_start is None: return
        dx = e.x - self.drag_start[0]
        dy = e.y - self.drag_start[1]
        self._pan_drag(dx, dy)
        self.drag_start = (e.x, e.y)

    def _orbit_drag(self, dx, dy):
        self.camera.yaw   += dx * 0.008
        self.camera.pitch += dy * 0.008
        self.camera.pitch = max(-math.pi/2+0.05, min(math.pi/2-0.05, self.camera.pitch))
        self.redraw()

    def _pan_drag(self, dx, dy):
        cam = self.camera
        eye = cam.position()
        f = vnorm(vsub(cam.target, eye))
        right = vnorm(vcross(f, [0,1,0]))
        up    = vcross(right, f)
        speed = cam.distance * 0.002
        cam.target = vadd(cam.target, vscale(right, -dx*speed))
        cam.target = vadd(cam.target, vscale(up,    dy*speed))
        self.redraw()

    def _vp_scroll(self, e):
        self._zoom(-1 if e.delta > 0 else 1)

    def _zoom(self, direction):
        self.camera.distance *= (1.12 if direction > 0 else 0.88)
        self.camera.distance = max(0.5, min(500, self.camera.distance))
        self.redraw()

    def _transform_drag(self, dx, dy, e):
        obj = self._get_selected()
        if obj is None: return
        sens = 0.015 * max(0.1, self.camera.distance / 8)
        if self.tool == 'move':
            if self.snap:
                dx = round(dx * sens / self.snap_val) * self.snap_val
                dy = round(dy * sens / self.snap_val) * self.snap_val
            else:
                dx *= sens; dy *= sens
            ax = self.trans_axis
            if ax == 'X':   obj['position'][0] += dx
            elif ax == 'Y': obj['position'][1] -= dy
            elif ax == 'Z': obj['position'][2] += dx
            else:
                obj['position'][0] += dx
                obj['position'][1] -= dy
        elif self.tool == 'rotate':
            angle = (dx + dy) * 0.02
            ax = self.trans_axis
            if ax == 'X':   obj['rotation'][0] += angle
            elif ax == 'Y': obj['rotation'][1] += angle
            elif ax == 'Z': obj['rotation'][2] += angle
            else:           obj['rotation'][1] += angle
        elif self.tool == 'scale':
            factor = 1 + (dx - dy) * 0.01
            factor = max(0.01, factor)
            ax = self.trans_axis
            if ax == 'X':   obj['scale'][0] *= factor
            elif ax == 'Y': obj['scale'][1] *= factor
            elif ax == 'Z': obj['scale'][2] *= factor
            else:
                obj['scale'][0] *= factor
                obj['scale'][1] *= factor
                obj['scale'][2] *= factor
        self.update_props()
        self.redraw()

    def _on_resize(self, e):
        self.camera.width  = e.width
        self.camera.height = e.height
        self.redraw()

    # ── PICKING ───────────────────────────────────────────────────────────────
    def _pick(self, mx, my):
        best_obj = None
        best_depth = float('inf')
        cam = self.camera
        cam.width  = self.canvas.winfo_width()
        cam.height = self.canvas.winfo_height()

        for obj in self.objects:
            if not obj['visible']: continue
            M = object_matrix(obj)
            for v in obj['verts']:
                wv = mat4_apply(M, v)
                sp = cam.project(wv)
                if sp is None: continue
                sx, sy, depth = sp
                if abs(sx - mx) < 12 and abs(sy - my) < 12:
                    if depth < best_depth:
                        best_depth = depth
                        best_obj = obj

        self.selected_id = best_obj['id'] if best_obj else None
        self.update_outliner()
        self.update_props()
        self.redraw()

    # ── KEYBOARD ──────────────────────────────────────────────────────────────
    def _on_key(self, e):
        key = e.keysym.lower()
        focused = str(self.root.focus_get())
        if 'entry' in focused or 'spinbox' in focused:
            return
        if key == 'q':          self.set_tool('select')
        elif key == 'g':        self.set_tool('move')
        elif key == 'r':        self.set_tool('rotate')
        elif key == 's':        self.set_tool('scale')
        elif key in ('x','delete'): self.delete_selected()
        elif key == 'd' and (e.state & 0x0001): self.duplicate_selected()
        elif key == 'a':        self._select_all()
        elif key == 'f':        self.focus_selected()
        elif key == 'z':        self.toggle_wireframe()
        elif key == 'h':        self.toggle_grid()
        elif key == 'kp_1':     self.set_view('front')
        elif key == 'kp_3':     self.set_view('right')
        elif key == 'kp_7':     self.set_view('top')
        elif key == 'kp_5':     self.set_view('persp')
        elif key == 'escape':   self.selected_id = None; self.update_outliner(); self.update_props(); self.redraw()
        elif key == 'z' and (e.state & 0x0004): self.undo()
        elif key == 'y' and (e.state & 0x0004): self.redo()
        # axis constraints during transform
        elif self.tool != 'select' and key in ('x','y','z'):
            self.trans_axis = key.upper()
            self.status_var.set(f'Constraint: {self.trans_axis} axis')

    # ── TOOLS ─────────────────────────────────────────────────────────────────
    def set_tool(self, tool):
        self.tool = tool
        self.trans_axis = 'XYZ'
        labels = {'select':'Select','move':'Move','rotate':'Rotate','scale':'Scale'}
        self.mode_var.set(f'Tool: {labels.get(tool,tool)}')
        self.status_var.set(f'{labels.get(tool,tool)}  [X/Y/Z to constrain axis]')
        self._highlight_tool()

    def _highlight_tool(self):
        for t, btn in self.tool_btns.items():
            if t == self.tool:
                btn.configure(bg='#e87d0d', fg='white')
            else:
                btn.configure(bg='#2a2a2a', fg='#aaa')

    def set_view(self, v):
        d = 10
        views = {
            'persp': (5,4,7,[0,0,0], 45, 25),
            'top':   (0,d,0,[0,0,0], 0, 89.9),
            'front': (0,0,d,[0,0,0], 0, 0),
            'right': (d,0,0,[0,0,0], -90, 0),
        }
        cfg = views.get(v, views['persp'])
        pos, tgt, yaw, pitch = cfg[0:3], cfg[3], cfg[4], cfg[5]
        cam = self.camera
        cam.target = list(tgt)
        cam.distance = d
        cam.yaw   = math.radians(yaw)
        cam.pitch = math.radians(pitch)
        for vid, btn in self.view_btns.items():
            btn.configure(bg='#e87d0d' if vid==v else '#2a2a2a',
                          fg='white' if vid==v else '#aaa')
        self.redraw()

    def toggle_wireframe(self):
        self.view_mode = 'wireframe' if self.view_mode == 'solid' else 'solid'
        self.status_var.set(f'View: {self.view_mode.capitalize()}')
        self.redraw()

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        self.redraw()

    def toggle_snap(self):
        self.snap = not self.snap
        self.snap_btn.configure(fg='#e87d0d' if self.snap else '#666')
        self.status_var.set(f'Snap: {"ON" if self.snap else "OFF"} ({self.snap_val})')

    # ── OBJECT OPS ────────────────────────────────────────────────────────────
    def _get_selected(self):
        if self.selected_id is None: return None
        for o in self.objects:
            if o['id'] == self.selected_id: return o
        return None

    def add_object(self, shape):
        count = sum(1 for o in self.objects if o['shape']==shape) + 1
        name  = f'{shape.capitalize()}.{count:03d}'
        obj   = make_object(shape, name)
        # offset from center
        angle = len(self.objects) * 0.7
        obj['position'] = [math.sin(angle)*0.3, 0.0, math.cos(angle)*0.3]
        self.objects.append(obj)
        self.selected_id = obj['id']
        self.history.snapshot()
        self.update_outliner()
        self.update_props()
        self.redraw()
        self.status_var.set(f'Added {name}')

    def delete_selected(self):
        obj = self._get_selected()
        if obj is None: return
        name = obj['name']
        self.objects = [o for o in self.objects if o['id'] != self.selected_id]
        self.selected_id = None
        self.history.snapshot()
        self.update_outliner()
        self.update_props()
        self.redraw()
        self.status_var.set(f'Deleted {name}')

    def duplicate_selected(self):
        obj = self._get_selected()
        if obj is None: return
        new_obj = copy.deepcopy(obj)
        new_obj['id'] = new_uid()
        new_obj['name'] = obj['name'] + '_copy'
        new_obj['position'] = [p+0.5 for p in obj['position']]
        self.objects.append(new_obj)
        self.selected_id = new_obj['id']
        self.history.snapshot()
        self.update_outliner()
        self.update_props()
        self.redraw()

    def _select_all(self):
        if self.objects:
            self.selected_id = self.objects[-1]['id']
            self.update_outliner()
            self.update_props()
            self.redraw()

    def focus_selected(self):
        obj = self._get_selected()
        if obj is None: return
        self.camera.target = list(obj['position'])
        self.redraw()

    def undo(self): self.history.undo()
    def redo(self): self.history.redo()

    # ── OUTLINER ──────────────────────────────────────────────────────────────
    def update_outliner(self):
        lb = self.outliner_list
        lb.delete(0, 'end')
        ICONS = {'cube':'■','sphere':'●','icosphere':'▲','cylinder':'⬛',
                 'cone':'▲','torus':'◎','plane':'▬'}
        for obj in self.objects:
            icon = ICONS.get(obj['shape'], '◆')
            lb.insert('end', f' {icon}  {obj["name"]}')
        # highlight selected
        for i, obj in enumerate(self.objects):
            if obj['id'] == self.selected_id:
                lb.selection_clear(0, 'end')
                lb.selection_set(i)
                lb.activate(i)

    def _on_outliner_select(self, e):
        sel = self.outliner_list.curselection()
        if not sel: return
        idx = sel[0]
        if 0 <= idx < len(self.objects):
            self.selected_id = self.objects[idx]['id']
            self.update_props()
            self.redraw()

    # ── PROPERTIES PANEL ──────────────────────────────────────────────────────
    def update_props(self):
        for w in self.props_frame.winfo_children():
            w.destroy()
        obj = self._get_selected()
        if obj is None:
            tk.Label(self.props_frame, text='No object selected',
                     bg='#252525', fg='#555', font=('Segoe UI',9,'italic'),
                     pady=12).pack(fill='x', padx=10)
            return

        self._props_section(obj)
        self.props_canvas.update_idletasks()
        self.props_canvas.configure(scrollregion=self.props_canvas.bbox('all'))

    def _props_section(self, obj):
        pad = {'padx': 8, 'pady': 2}

        # Object name
        tk.Label(self.props_frame, text='Name', bg='#252525', fg='#888',
                 font=('Segoe UI',8), anchor='w').pack(fill='x', **pad)
        name_var = tk.StringVar(value=obj['name'])
        name_e = tk.Entry(self.props_frame, textvariable=name_var,
                          bg='#1e1e1e', fg='white', relief='flat',
                          insertbackground='white', font=('Segoe UI',9))
        name_e.pack(fill='x', padx=8, pady=(0,6))
        def on_name(*_):
            obj['name'] = name_var.get()
            self.update_outliner()
        name_var.trace_add('write', on_name)

        # Transform
        self._section_label('Transform')
        axes_labels = ['X', 'Y', 'Z']
        for prop, key in [('Location','position'),('Rotation','rotation'),('Scale','scale')]:
            row = tk.Frame(self.props_frame, bg='#252525')
            row.pack(fill='x', padx=8, pady=1)
            tk.Label(row, text=prop, bg='#252525', fg='#888',
                     font=('Segoe UI',8), width=7, anchor='w').pack(side='left')
            axis_colors = ['#c44','#4a4','#46c']
            for i, (ax, col) in enumerate(zip(axes_labels, axis_colors)):
                tk.Label(row, text=ax, bg='#252525', fg=col,
                         font=('Segoe UI',8,'bold')).pack(side='left')
                var = tk.DoubleVar(value=round(obj[key][i],4))
                sp = tk.Spinbox(row, textvariable=var, from_=-9999, to=9999,
                                increment=0.1 if prop!='Rotation' else 1,
                                width=6, bg='#1e1e1e', fg='#ddd',
                                buttonbackground='#333', relief='flat',
                                font=('Consolas',8))
                sp.pack(side='left', padx=1)
                def on_change(v=var, k=key, idx=i, o=obj):
                    try:
                        o[k][idx] = float(v.get())
                        self.redraw()
                    except: pass
                var.trace_add('write', lambda *_, fn=on_change: fn())

        # Material
        self._section_label('Material')
        color_frame = tk.Frame(self.props_frame, bg='#252525')
        color_frame.pack(fill='x', padx=8, pady=2)
        tk.Label(color_frame, text='Color', bg='#252525', fg='#888',
                 font=('Segoe UI',8), width=7, anchor='w').pack(side='left')
        color_swatch = tk.Frame(color_frame, bg=obj['color'], width=80, height=18,
                                cursor='hand2', relief='flat')
        color_swatch.pack(side='left')
        def pick_color():
            result = colorchooser.askcolor(color=obj['color'], parent=self.root)
            if result[1]:
                obj['color'] = result[1]
                color_swatch.configure(bg=obj['color'])
                self.redraw()
        color_swatch.bind('<Button-1>', lambda e: pick_color())

        # Visibility + Wireframe
        vis_frame = tk.Frame(self.props_frame, bg='#252525')
        vis_frame.pack(fill='x', padx=8, pady=2)
        vis_var = tk.BooleanVar(value=obj['visible'])
        wf_var  = tk.BooleanVar(value=obj['wireframe'])
        def on_vis():
            obj['visible'] = vis_var.get(); self.redraw()
        def on_wf():
            obj['wireframe'] = wf_var.get(); self.redraw()
        tk.Checkbutton(vis_frame, text='Visible', variable=vis_var, command=on_vis,
                       bg='#252525', fg='#aaa', selectcolor='#1e1e1e',
                       activebackground='#252525', font=('Segoe UI',8)).pack(side='left')
        tk.Checkbutton(vis_frame, text='Wireframe', variable=wf_var, command=on_wf,
                       bg='#252525', fg='#aaa', selectcolor='#1e1e1e',
                       activebackground='#252525', font=('Segoe UI',8)).pack(side='left', padx=8)

        # Delete / Duplicate buttons
        btn_row = tk.Frame(self.props_frame, bg='#252525')
        btn_row.pack(fill='x', padx=8, pady=6)
        tk.Button(btn_row, text='Duplicate [Shift+D]', bg='#3a3a3a', fg='#aaa',
                  relief='flat', bd=0, padx=6, pady=4, cursor='hand2',
                  font=('Segoe UI',8), command=self.duplicate_selected).pack(side='left', fill='x', expand=True)
        tk.Button(btn_row, text='Delete [X]', bg='#7a2020', fg='#fff',
                  relief='flat', bd=0, padx=6, pady=4, cursor='hand2',
                  font=('Segoe UI',8), command=self.delete_selected).pack(side='left', fill='x', expand=True, padx=(4,0))

    def _section_label(self, text):
        f = tk.Frame(self.props_frame, bg='#2a2a2a')
        f.pack(fill='x', pady=(6,2))
        tk.Label(f, text=text, bg='#2a2a2a', fg='#888',
                 font=('Segoe UI', 8, 'bold'), anchor='w',
                 padx=8, pady=3).pack(fill='x')

    # ── RENDERING ─────────────────────────────────────────────────────────────
    def redraw(self):
        c = self.canvas
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 2 or h < 2: return
        c.delete('all')
        self.camera.width  = w
        self.camera.height = h

        # Background gradient (fake)
        c.create_rectangle(0, 0, w, h, fill='#1c1c1c', outline='')

        if self.show_grid:
            self._draw_grid(c)

        # Collect all visible faces with depth
        draw_list = []
        for obj in self.objects:
            if not obj['visible']: continue
            M = object_matrix(obj)
            # project verts
            proj = []
            for v in obj['verts']:
                wv = mat4_apply(M, v)
                sp = self.camera.project(wv)
                proj.append(sp)

            # Faces
            wf = obj['wireframe'] or (self.view_mode == 'wireframe')
            for face in obj['faces']:
                pts = []
                ok = True
                for vi in face:
                    if proj[vi] is None: ok=False; break
                    pts.append((proj[vi][0], proj[vi][1]))
                if not ok or len(pts)<3: continue
                avg_depth = sum(proj[vi][2] for vi in face) / len(face)
                # face normal
                wverts = [mat4_apply(M, obj['verts'][vi]) for vi in face[:3]]
                fn = face_normal(wverts)
                eye = self.camera.position()
                center = [sum(wverts[i][j] for i in range(3))/3 for j in range(3)]
                to_eye = vsub(eye, center)
                facing = vdot(fn, to_eye) > 0
                draw_list.append({
                    'type': 'face', 'pts': pts, 'obj': obj,
                    'depth': avg_depth, 'facing': facing,
                    'normal': fn, 'wireframe': wf,
                })

            # Edges (wireframe or selection highlight)
            selected = obj['id'] == self.selected_id
            if wf or selected:
                for a, b in obj['edges']:
                    if proj[a] is None or proj[b] is None: continue
                    avg_depth = (proj[a][2]+proj[b][2])/2
                    draw_list.append({
                        'type': 'edge',
                        'pts': [(proj[a][0],proj[a][1]),(proj[b][0],proj[b][1])],
                        'obj': obj, 'depth': avg_depth, 'selected': selected,
                        'wireframe': wf,
                    })

        # Sort back-to-front
        draw_list.sort(key=lambda x: -x['depth'])

        # Build live light direction from sliders
        az  = math.radians(self.light_azimuth.get())
        el  = math.radians(self.light_elevation.get())
        ldir = [math.cos(el)*math.sin(az), math.sin(el), math.cos(el)*math.cos(az)]
        lint  = self.light_intensity.get()
        lamb  = self.light_ambient.get()
        lcol  = hex_to_rgb_tuple(self.light_color)

        for item in draw_list:
            if item['type'] == 'face':
                obj = item['obj']
                wf  = item['wireframe']
                if wf:
                    c.create_polygon(item['pts'], fill='', outline=obj['color'], width=1)
                else:
                    if item['facing']:
                        col = shade_full(obj['color'], item['normal'], ldir, lint, lamb, lcol)
                    else:
                        col = darken(obj['color'], 0.35)
                    selected = obj['id'] == self.selected_id
                    outline = '#e87d0d' if selected else ''
                    c.create_polygon(item['pts'], fill=col,
                                     outline=outline, width=1 if selected else 0)
            elif item['type'] == 'edge':
                obj = item['obj']
                wf  = item['wireframe']
                sel = item['selected']
                if sel and not wf:
                    col = '#e87d0d'; w2 = 1
                elif wf:
                    col = obj['color']; w2 = 1
                else:
                    col = '#e87d0d'; w2 = 1
                x1,y1 = item['pts'][0]; x2,y2 = item['pts'][1]
                c.create_line(x1,y1,x2,y2, fill=col, width=w2)

        # Origin dot for selected
        sel_obj = self._get_selected()
        if sel_obj:
            sp = self.camera.project(sel_obj['position'])
            if sp:
                x, y = sp[0], sp[1]
                c.create_oval(x-4,y-4,x+4,y+4, fill='#e87d0d', outline='white', width=1)

        # Axis gizmo (bottom-left)
        self._draw_axis_gizmo(c, 50, h-50)

        # View label
        view_names = {'persp':'Perspective','top':'Top Ortho',
                      'front':'Front Ortho','right':'Right Ortho'}
        c.create_text(10, 10, text=view_names.get(
            self._current_view(), 'Perspective'),
            anchor='nw', fill='#888', font=('Segoe UI', 9))

        # Tool/axis hint
        if self.tool != 'select' and self.trans_axis != 'XYZ':
            c.create_text(w//2, 30,
                          text=f'{self.tool.upper()} · Axis: {self.trans_axis}',
                          fill='#e87d0d', font=('Segoe UI', 11, 'bold'))

        self._frames += 1

    def _draw_grid(self, c):
        cam = self.camera
        w, h = cam.width, cam.height
        size = 10
        step = 1
        for i in range(-size, size+1):
            # X lines
            p1 = cam.project([i*step, 0, -size*step])
            p2 = cam.project([i*step, 0,  size*step])
            if p1 and p2:
                col = '#3a3a3a' if i != 0 else '#555'
                c.create_line(p1[0],p1[1],p2[0],p2[1], fill=col, width=1)
            # Z lines
            p1 = cam.project([-size*step, 0, i*step])
            p2 = cam.project([ size*step, 0, i*step])
            if p1 and p2:
                col = '#3a3a3a' if i != 0 else '#555'
                c.create_line(p1[0],p1[1],p2[0],p2[1], fill=col, width=1)

    def _draw_axis_gizmo(self, c, cx, cy):
        cam = self.camera
        eye = cam.position()
        t   = cam.target
        axes = [(1,0,0,'#e44','X'),(0,1,0,'#4c4','Y'),(0,0,1,'#46c','Z')]
        r = 30
        for dx,dy,dz,col,lbl in axes:
            wpt = [t[0]+dx*2, t[1]+dy*2, t[2]+dz*2]
            sp  = cam.project(wpt)
            if sp:
                nx = cx + (sp[0]-cam.width//2)/cam.width * r * 4
                ny = cy + (sp[1]-cam.height//2)/cam.height * r * 4
                c.create_line(cx, cy, nx, ny, fill=col, width=2)
                c.create_oval(nx-5,ny-5,nx+5,ny+5, fill=col, outline='')
                c.create_text(nx, ny, text=lbl, fill='white',
                              font=('Segoe UI',7,'bold'))

    def _current_view(self):
        for v, btn in self.view_btns.items():
            if btn.cget('bg') == '#e87d0d':
                return v
        return 'persp'

    # ── FPS ───────────────────────────────────────────────────────────────────
    def _schedule_fps(self):
        def update():
            now = time.time()
            elapsed = now - self._last_time
            if elapsed >= 1.0:
                fps = self._frames / elapsed
                self.fps_label.configure(text=f'FPS: {fps:.0f}')
                self._frames = 0
                self._last_time = now
            self.root.after(500, update)
        self.root.after(500, update)

    # ── EXPORT ────────────────────────────────────────────────────────────────
    def export_scene(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[
                ('3D Creator Scene (JSON)', '*.json'),
                ('Wavefront OBJ',           '*.obj'),
                ('All files',               '*.*'),
            ],
            title='Export Scene')
        if not path: return
        if path.lower().endswith('.obj'):
            self._export_obj(path)
        else:
            self._export_json(path)

    def _export_json(self, path):
        data = []
        for o in self.objects:
            data.append({k: v for k, v in o.items()
                         if k not in ('verts', 'edges', 'faces')})
        payload = {
            'version':  '1.0',
            'app':      '3D Creator',
            'exported': time.strftime('%Y-%m-%d %H:%M:%S'),
            'objects':  data,
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        messagebox.showinfo('Export', f'Scene saved to:\n{os.path.basename(path)}')

    def _export_obj(self, path):
        lines = ['# 3D Creator Export', f'# {time.strftime("%Y-%m-%d %H:%M:%S")}']
        vtx_offset = 1
        for obj in self.objects:
            if not obj['visible']: continue
            r, g, b = hex_to_rgb_tuple(obj['color'])
            lines += [
                f'\no {obj["name"]}',
                f'# color {obj["color"]}',
                f'usemtl {obj["name"]}_mat',
            ]
            M = object_matrix(obj)
            for v in obj['verts']:
                wv = mat4_apply(M, v)
                lines.append(f'v {wv[0]:.4f} {wv[1]:.4f} {wv[2]:.4f}')
            for face in obj['faces']:
                idxs = ' '.join(str(i + vtx_offset) for i in face)
                lines.append(f'f {idxs}')
            vtx_offset += len(obj['verts'])
        # Write companion .mtl
        mtl_path = path.replace('.obj', '.mtl')
        mtl_lines = ['# 3D Creator Materials']
        for obj in self.objects:
            if not obj['visible']: continue
            r, g, b = hex_to_rgb_tuple(obj['color'])
            mtl_lines += [
                f'\nnewmtl {obj["name"]}_mat',
                f'Kd {r/255:.4f} {g/255:.4f} {b/255:.4f}',
                'Ka 0.2 0.2 0.2',
                'Ks 0.1 0.1 0.1',
                'Ns 32',
            ]
        with open(path, 'w') as f:
            f.write(f'mtllib {os.path.basename(mtl_path)}\n')
            f.write('\n'.join(lines))
        with open(mtl_path, 'w') as f:
            f.write('\n'.join(mtl_lines))
        messagebox.showinfo('Export',
            f'OBJ saved to:\n{os.path.basename(path)}\n'
            f'Materials:  {os.path.basename(mtl_path)}')

    # ── IMPORT ────────────────────────────────────────────────────────────────
    def import_scene(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ('3D Creator Scene (JSON)', '*.json'),
                ('Wavefront OBJ',           '*.obj'),
                ('All files',               '*.*'),
            ],
            title='Import Scene')
        if not path: return
        if path.lower().endswith('.obj'):
            self._import_obj(path)
        else:
            self._import_json(path)

    def _import_json(self, path):
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception as e:
            messagebox.showerror('Import Error', f'Could not read file:\n{e}')
            return

        raw_objects = payload.get('objects', payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_objects, list):
            messagebox.showerror('Import Error', 'Invalid scene file format.')
            return

        choice = messagebox.askyesnocancel(
            'Import Scene',
            'Replace current scene, or merge into it?\n\n'
            'Yes = Replace\nNo = Merge\nCancel = Abort')
        if choice is None: return
        if choice:
            self.objects.clear()
            self.selected_id = None

        loaded = 0
        for d in raw_objects:
            shape = d.get('shape', 'cube')
            gen   = MESH_GEN.get(shape, make_cube)
            verts, edges, faces = gen()
            obj = {
                'id':       new_uid(),
                'name':     d.get('name', shape.capitalize()),
                'shape':    shape,
                'verts':    verts,
                'edges':    edges,
                'faces':    faces,
                'position': d.get('position', [0.0, 0.0, 0.0]),
                'rotation': d.get('rotation', [0.0, 0.0, 0.0]),
                'scale':    d.get('scale',    [1.0, 1.0, 1.0]),
                'color':    d.get('color',    random_color()),
                'visible':  d.get('visible',  True),
                'wireframe':d.get('wireframe',False),
            }
            self.objects.append(obj)
            loaded += 1

        if self.objects:
            self.selected_id = self.objects[-1]['id']
        self.history.snapshot()
        self.update_outliner()
        self.update_props()
        self.redraw()
        self.status_var.set(f'Imported {loaded} object(s) from {os.path.basename(path)}')

    def _import_obj(self, path):
        """Basic OBJ parser — imports vertices and faces as a single mesh per object group."""
        try:
            with open(path) as f:
                lines = f.readlines()
        except Exception as e:
            messagebox.showerror('Import Error', str(e))
            return

        choice = messagebox.askyesnocancel(
            'Import OBJ',
            'Replace current scene, or merge?\n\nYes = Replace   No = Merge   Cancel = Abort')
        if choice is None: return
        if choice:
            self.objects.clear()
            self.selected_id = None

        groups   = {}
        cur_name = 'Imported'
        all_verts= []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if parts[0] == 'o':
                cur_name = parts[1] if len(parts) > 1 else 'Object'
                if cur_name not in groups:
                    groups[cur_name] = {'faces': [], 'color': random_color()}
            elif parts[0] == 'v':
                try:
                    all_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (IndexError, ValueError):
                    pass
            elif parts[0] == 'f':
                if cur_name not in groups:
                    groups[cur_name] = {'faces': [], 'color': random_color()}
                face = []
                for tok in parts[1:]:
                    idx = int(tok.split('/')[0])
                    face.append((idx - 1) if idx > 0 else (len(all_verts) + idx))
                groups[cur_name]['faces'].append(face)

        if not groups or not all_verts:
            messagebox.showerror('Import Error', 'No geometry found in OBJ file.')
            return

        loaded = 0
        for gname, gdata in groups.items():
            if not gdata['faces']: continue
            used_idx  = sorted({i for f in gdata['faces'] for i in f})
            idx_map   = {old: new for new, old in enumerate(used_idx)}
            verts     = [all_verts[i] for i in used_idx if i < len(all_verts)]
            faces     = [[idx_map[i] for i in f if i in idx_map] for f in gdata['faces']]
            faces     = [f for f in faces if len(f) >= 3]
            edges_set = set()
            for face in faces:
                for k in range(len(face)):
                    a, b = face[k], face[(k+1) % len(face)]
                    edges_set.add((min(a,b), max(a,b)))

            # Centre the mesh
            if verts:
                cx = sum(v[0] for v in verts) / len(verts)
                cy = sum(v[1] for v in verts) / len(verts)
                cz = sum(v[2] for v in verts) / len(verts)
                verts = [[v[0]-cx, v[1]-cy, v[2]-cz] for v in verts]
                pos = [cx, cy, cz]
            else:
                pos = [0.0, 0.0, 0.0]

            obj = {
                'id':       new_uid(),
                'name':     gname,
                'shape':    'cube',
                'verts':    verts,
                'edges':    list(edges_set),
                'faces':    faces,
                'position': pos,
                'rotation': [0.0, 0.0, 0.0],
                'scale':    [1.0, 1.0, 1.0],
                'color':    gdata['color'],
                'visible':  True,
                'wireframe':False,
            }
            self.objects.append(obj)
            loaded += 1

        if self.objects:
            self.selected_id = self.objects[-1]['id']
        self.history.snapshot()
        self.update_outliner()
        self.update_props()
        self.redraw()
        self.status_var.set(f'Imported {loaded} OBJ group(s) from {os.path.basename(path)}')

    # ── AI ASSISTANT ──────────────────────────────────────────────────────────
    def open_ai_panel(self):
        if hasattr(self, '_ai_win') and self._ai_win.winfo_exists():
            self._ai_win.lift()
            return

        win = tk.Toplevel(self.root)
        win.title('AI Assistant — Llama')
        win.geometry('480x620')
        win.configure(bg='#1a1a1a')
        win.resizable(True, True)
        self._ai_win = win

        # ── Header
        hdr = tk.Frame(win, bg='#2a2a2a', height=42)
        hdr.pack(fill='x')
        hdr.pack_propagate(False)
        tk.Label(hdr, text=' 🦙 Llama AI Assistant', bg='#2a2a2a', fg='#e87d0d',
                 font=('Segoe UI', 11, 'bold')).pack(side='left', padx=10)

        # Model selector
        self._ai_model = tk.StringVar(value='llama3')
        models = ['llama3','llama3.2','llama3.1','llama2','mistral','codellama','phi3']
        tk.Label(hdr, text='Model:', bg='#2a2a2a', fg='#888',
                 font=('Segoe UI', 9)).pack(side='right', padx=(0,4))
        model_cb = ttk.Combobox(hdr, textvariable=self._ai_model,
                                values=models, width=12, state='normal')
        model_cb.pack(side='right', padx=4)

        # ── Status bar
        self._ai_status_var = tk.StringVar(value='Checking Ollama...')
        status_bar = tk.Label(win, textvariable=self._ai_status_var,
                              bg='#111', fg='#666', font=('Consolas', 8),
                              anchor='w', padx=8, pady=3)
        status_bar.pack(fill='x')

        # ── Chat history
        chat_frame = tk.Frame(win, bg='#1a1a1a')
        chat_frame.pack(fill='both', expand=True, padx=6, pady=4)

        self._chat_text = tk.Text(chat_frame, bg='#1a1a1a', fg='#ccc',
                                  font=('Segoe UI', 10), wrap='word',
                                  relief='flat', state='disabled',
                                  insertbackground='white',
                                  selectbackground='#e87d0d33',
                                  padx=8, pady=6)
        chat_scroll = tk.Scrollbar(chat_frame, command=self._chat_text.yview,
                                   bg='#2a2a2a', troughcolor='#111')
        self._chat_text.configure(yscrollcommand=chat_scroll.set)
        chat_scroll.pack(side='right', fill='y')
        self._chat_text.pack(fill='both', expand=True)

        # Text tags
        self._chat_text.tag_configure('user',   foreground='#e87d0d', font=('Segoe UI',10,'bold'))
        self._chat_text.tag_configure('ai',     foreground='#8bcfff', font=('Segoe UI',10))
        self._chat_text.tag_configure('system', foreground='#666',    font=('Segoe UI', 9,'italic'))
        self._chat_text.tag_configure('error',  foreground='#e44',    font=('Segoe UI', 9,'italic'))

        # ── Scene context checkbox
        ctx_row = tk.Frame(win, bg='#1e1e1e')
        ctx_row.pack(fill='x', padx=6, pady=(0,2))
        self._send_ctx = tk.BooleanVar(value=True)
        tk.Checkbutton(ctx_row, text='Include scene context', variable=self._send_ctx,
                       bg='#1e1e1e', fg='#888', selectcolor='#1a1a1a',
                       activebackground='#1e1e1e', font=('Segoe UI', 8)).pack(side='left')
        tk.Button(ctx_row, text='Clear chat', bg='#1e1e1e', fg='#666',
                  relief='flat', bd=0, font=('Segoe UI', 8), cursor='hand2',
                  command=self._clear_chat).pack(side='right', padx=6)

        # ── Input area
        input_frame = tk.Frame(win, bg='#2a2a2a', pady=6)
        input_frame.pack(fill='x', padx=6, pady=(0,6))

        self._ai_input = tk.Text(input_frame, height=3, bg='#1e1e1e', fg='#ddd',
                                 font=('Segoe UI', 10), relief='flat',
                                 insertbackground='white', wrap='word',
                                 padx=6, pady=4)
        self._ai_input.pack(fill='x', side='left', expand=True)
        self._ai_input.bind('<Return>', self._ai_send_enter)
        self._ai_input.bind('<Shift-Return>', lambda e: None)

        self._send_btn = tk.Button(input_frame, text='Send\n[↵]',
                                   bg='#e87d0d', fg='white', relief='flat',
                                   bd=0, padx=10, font=('Segoe UI', 9, 'bold'),
                                   cursor='hand2', command=self._ai_send)
        self._send_btn.pack(side='right', fill='y', padx=(4,0))

        # Welcome message
        self._chat_append('system',
            'Ask Llama anything about 3D modeling, scene composition, shortcuts, and more.\n'
            'Examples:\n'
            '  • "How do I make a realistic rock?\"\n'
            '  • "What objects should I add for a city scene?\"\n'
            '  • "How do I fix lighting that looks flat?\"\n\n')

        # Check Ollama in background
        threading.Thread(target=self._check_ollama, daemon=True).start()

    def _check_ollama(self):
        try:
            req = urllib.request.urlopen('http://localhost:11434/api/tags', timeout=3)
            data = json.loads(req.read())
            models = [m['name'] for m in data.get('models', [])]
            if models:
                self._ai_status('Connected  |  Available: ' + ', '.join(models[:5]))
                # update combobox with real models
                self.root.after(0, lambda: None)
            else:
                self._ai_status('Ollama connected but no models pulled yet.')
                self.root.after(0, lambda: self._chat_append('error',
                    '⚠ Ollama is running but no models are installed.\n'
                    'Run in terminal:\n  ollama pull llama3\n\n'))
        except Exception:
            self._ai_status('Ollama not found — see setup instructions below.')
            self.root.after(0, self._show_setup_instructions)

    def _show_setup_instructions(self):
        self._chat_append('error',
            '⚠ Ollama is not running or not installed.\n\n')
        self._chat_append('system',
            'Quick setup (takes ~2 min):\n\n'
            '  1. Install Ollama:\n'
            '     https://ollama.com/download\n\n'
            '  2. Open Terminal and run:\n'
            '     ollama pull llama3\n\n'
            '  3. Ollama starts automatically in the background.\n\n'
            '  4. Reopen this panel and ask away!\n\n'
            'Once running, Llama will give you expert 3D advice\n'
            'right inside the app, with your scene as context.\n\n')

    def _ai_status(self, msg):
        self.root.after(0, lambda: self._ai_status_var.set(msg))

    def _chat_append(self, tag, text):
        t = self._chat_text
        t.configure(state='normal')
        t.insert('end', text, tag)
        t.configure(state='disabled')
        t.see('end')

    def _clear_chat(self):
        self._chat_text.configure(state='normal')
        self._chat_text.delete('1.0', 'end')
        self._chat_text.configure(state='disabled')

    def _ai_send_enter(self, e):
        if not (e.state & 0x0001):   # Shift not held
            self._ai_send()
            return 'break'

    def _ai_send(self):
        prompt = self._ai_input.get('1.0', 'end').strip()
        if not prompt:
            return
        self._ai_input.delete('1.0', 'end')
        self._send_btn.configure(state='disabled', text='...')

        # Build scene context
        ctx = ''
        if self._send_ctx.get():
            ctx = self._build_scene_context()

        self._chat_append('user', f'You: {prompt}\n\n')
        self._chat_append('ai', 'Llama: ')

        threading.Thread(target=self._query_ollama,
                         args=(prompt, ctx), daemon=True).start()

    def _build_scene_context(self):
        lines = ['Current 3D scene:']
        for o in self.objects:
            sel = ' (SELECTED)' if o['id'] == self.selected_id else ''
            px,py,pz = o['position']
            lines.append(f'  - {o["name"]} ({o["shape"]}) at ({px:.1f},{py:.1f},{pz:.1f}){sel}')
        lines.append(f'Total objects: {len(self.objects)}')
        lines.append(f'Active tool: {self.tool}')
        return '\n'.join(lines)

    def _query_ollama(self, prompt, ctx):
        model = self._ai_model.get() or 'llama3'
        system = (
            'You are an expert 3D artist and Blender/3D modeling assistant embedded inside a '
            '3D Creator app. Give concise, practical advice. When the user asks what to build '
            'or how to improve their scene, give specific, actionable steps. '
            'Keep answers focused and under 200 words unless asked for more detail.'
        )
        full_prompt = f'{ctx}\n\nUser question: {prompt}' if ctx else prompt

        payload = json.dumps({
            'model':  model,
            'prompt': full_prompt,
            'system': system,
            'stream': True,
        }).encode()

        try:
            req = urllib.request.Request(
                'http://localhost:11434/api/generate',
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST')
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line.decode())
                        token = chunk.get('response', '')
                        if token:
                            self.root.after(0, lambda t=token: self._stream_token(t))
                        if chunk.get('done'):
                            break
                    except json.JSONDecodeError:
                        continue
            self.root.after(0, lambda: self._chat_append('ai', '\n\n'))
        except urllib.error.URLError:
            self.root.after(0, lambda: self._chat_append('error',
                '\n[Ollama not reachable. Is it running? See setup above.]\n\n'))
        except Exception as e:
            self.root.after(0, lambda: self._chat_append('error',
                f'\n[Error: {e}]\n\n'))
        finally:
            self.root.after(0, self._ai_done)

    def _stream_token(self, token):
        self._chat_text.configure(state='normal')
        self._chat_text.insert('end', token, 'ai')
        self._chat_text.configure(state='disabled')
        self._chat_text.see('end')

    def _ai_done(self):
        self._send_btn.configure(state='normal', text='Send\n[↵]')
        self._ai_status('Ready')

    # ── RUN ───────────────────────────────────────────────────────────────────
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = App()
    app.run()
