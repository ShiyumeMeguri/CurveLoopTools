import bpy
import bmesh
import mathutils
import math
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty
from bpy.types import Operator

# ########################################
# ##### General Math functions ###########
# ########################################

def calculate_cubic_splines(tknots, knots):
    """
    Calculates natural cubic splines through all given knots.
    Adapted from LoopTools to support N-dimensions.
    """
    n = len(knots)
    if n < 2:
        return False
    
    # Check dimension
    if isinstance(knots[0], (float, int)):
        dim = 1
        locs = [[k] for k in knots]
    else:
        dim = len(knots[0])
        locs = [list(k) for k in knots]
        
    x = tknots[:]
    result = []
    
    # Solve for each dimension independently
    for j in range(dim):
        a = []
        for i in locs:
            a.append(i[j])
        h = []
        for i in range(n - 1):
            val = x[i + 1] - x[i]
            if val == 0:
                h.append(1e-8)
            else:
                h.append(val)
        
        q = [0.0]
        for i in range(1, n - 1):
            term = 3 / h[i] * (a[i + 1] - a[i]) - 3 / h[i - 1] * (a[i] - a[i - 1])
            q.append(term)
            
        l = [1.0]
        u = [0.0]
        z = [0.0]
        
        for i in range(1, n - 1):
            val = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1]
            if val == 0:
                val = 1e-8
            l.append(val)
            u.append(h[i] / l[i])
            z.append((q[i] - h[i - 1] * z[i - 1]) / l[i])
            
        l.append(1.0)
        z.append(0.0)
        
        b = [0.0 for _ in range(n - 1)]
        c = [0.0 for _ in range(n)]
        d = [0.0 for _ in range(n - 1)]
        
        c[n - 1] = 0.0
        for i in range(n - 2, -1, -1):
            c[i] = z[i] - u[i] * c[i + 1]
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
            
        for i in range(n - 1):
            if len(result) <= i:
                result.append([])
            result[i].append([a[i], b[i], c[i], d[i], x[i]])

    splines = []
    for i in range(n - 1):
        splines.append(result[i])
        
    return splines

def calculate_linear_splines(tknots, knots):
    """
    Calculates linear splines.
    """
    splines = []
    if isinstance(knots[0], (float, int)):
        dim = 1
        conversion = lambda x: [x]
    else:
        dim = len(knots[0])
        conversion = lambda x: list(x)
        
    for i in range(len(knots) - 1):
        a = conversion(knots[i])
        b = conversion(knots[i + 1])
        d = [b[k] - a[k] for k in range(dim)]
        t = tknots[i]
        u = tknots[i + 1] - t
        segment_splines = []
        for k in range(dim):
            segment_splines.append([a[k], d[k], t, u])
        splines.append(segment_splines)
        
    return splines

# ########################################
# ##### Relax logic ######################
# ########################################

def relax_calculate_knots(points_len, circular):
    knots = [[], []]
    points = [[], []]
    loop = list(range(points_len))
    
    if circular:
        if len(loop) % 2 == 1: extend = [False, True, 0, 1, 0, 1]
        else: extend = [True, False, 0, 1, 1, 2]
    else:
        extend = [False, False, 0, 1, 1, 2]
             
    for j in range(2):
        temp_loop = loop[:]
        if extend[j]: temp_loop = [loop[-1]] + loop + [loop[0]]
        k_indices = []
        for i in range(extend[2 + 2 * j], len(temp_loop), 2):
            k_indices.append(temp_loop[i])
        knots[j] = k_indices
        p_indices = []
        for i in range(extend[3 + 2 * j], len(temp_loop), 2):
            idx = temp_loop[i]
            if idx == loop[-1] and not circular: continue
            if len(p_indices) == 0 or idx != p_indices[0]:
                p_indices.append(idx)
        points[j] = p_indices
        if circular and knots[j][0] != knots[j][-1]:
            knots[j].append(knots[j][0])
            
    if len(points[1]) == 0:
        knots.pop(1)
        points.pop(1)
        
    return knots, points

def relax_calculate_t(points_co, knots, points_indices, regular):
    all_tknots = []
    all_tpoints = []
    for i in range(len(knots)):
        k_list, p_list = knots[i], points_indices[i]
        mix = []
        nk, np = len(k_list), len(p_list)
        max_len = max(nk, np)
        for j in range(max_len):
            if j < nk: mix.append((True, k_list[j]))
            if j < np: mix.append((False, p_list[j]))
        len_total, loc_prev, tknots, tpoints = 0, None, [], []
        for is_knot, idx in mix:
            loc = mathutils.Vector(points_co[idx])
            if loc_prev is None: loc_prev = loc
            len_total += (loc - loc_prev).length
            if is_knot: tknots.append(len_total)
            else: tpoints.append(len_total)
            loc_prev = loc
        if regular:
            new_tpoints = []
            for p_idx in range(len(tpoints)):
                if p_idx + 1 < len(tknots):
                    new_tpoints.append((tknots[p_idx] + tknots[p_idx+1]) / 2.0)
                else:
                    new_tpoints.append(tpoints[p_idx])
            tpoints = new_tpoints
        all_tknots.append(tknots)
        all_tpoints.append(tpoints)
    return all_tknots, all_tpoints

def relax_calculate_verts(interpolation, tknots, knots, tpoints, points_indices, splines):
    moves = []
    for i in range(len(knots)):
        p_list, tk, tp, seg_splines = points_indices[i], tknots[i], tpoints[i], splines[i]
        for j, p_idx in enumerate(p_list):
            if j >= len(tp): continue
            m, n = tp[j], -1
            if m in tk: n = tk.index(m)
            else:
                for k_idx in range(len(tk)):
                    if tk[k_idx] > m:
                        n = k_idx - 1
                        break
                if n == -1: n = len(tk) - 1
            n = max(0, min(n, len(seg_splines) - 1))
            new_vals = []
            if interpolation == 'cubic':
                dims = seg_splines[n]
                for d_idx in range(len(dims)):
                    a, b, c, d_coeff, tx = dims[d_idx]
                    dt = m - tx
                    new_vals.append(a + b*dt + c*(dt**2) + d_coeff*(dt**3))
            else:
                dims = seg_splines[n]
                for d_idx in range(len(dims)):
                    a, d_val, t, u = dims[d_idx]
                    if u == 0: u = 1e-8
                    new_vals.append(((m - t) / u) * d_val + a)
            moves.append((p_idx, new_vals))
    return moves

# ########################################
# ##### Space logic ######################
# ########################################

def space_calculate_t(points_co):
    tknots, loc_prev, len_total = [], None, 0
    for loc in points_co:
        loc = mathutils.Vector(loc)
        if loc_prev is None: loc_prev = loc
        len_total += (loc - loc_prev).length
        tknots.append(len_total)
        loc_prev = loc
    amount = len(points_co)
    if amount < 2: return tknots, tknots
    t_per_segment = len_total / (amount - 1)
    tpoints = [i * t_per_segment for i in range(amount)]
    return tknots, tpoints

def space_calculate_verts(interpolation, tknots, tpoints, splines):
    moves = []
    for i, m in enumerate(tpoints):
        n = -1
        for k_idx in range(len(tknots)-1):
            if tknots[k_idx] <= m <= tknots[k_idx+1]:
                n = k_idx
                break
        if n == -1:
            if m <= tknots[0]: n = 0
            elif m >= tknots[-1]: n = len(tknots) - 2
        n = max(0, min(n, len(splines) - 1))
        new_vals = []
        if interpolation == 'cubic':
            dims = splines[n]
            for d_idx in range(len(dims)):
                a, b, c, d_coeff, tx = dims[d_idx]
                dt = m - tx
                new_vals.append(a + b*dt + c*(dt**2) + d_coeff*(dt**3))
        else:
            dims = splines[n]
            for d_idx in range(len(dims)):
                a, d_val, t, u = dims[d_idx]
                if u == 0: u = 1e-8
                new_vals.append(((m - t) / u) * d_val + a)
        moves.append((i, new_vals))
    return moves

# ########################################
# ##### Curve Operators ##################
# ########################################

class LOOPTOOLSPLUS_OT_curve_relax(Operator):
    bl_idname = "looptools_plus.curve_relax"
    bl_label = "Relax"
    bl_description = "Relax the curve, smoothing it out"
    bl_options = {'REGISTER', 'UNDO'}

    relax_position: BoolProperty(name="Relax Position", default=True)
    relax_tilt: BoolProperty(name="Relax Tilt", default=False)
    relax_radius: BoolProperty(name="Relax Radius", default=False)
    opt_lock_length: BoolProperty(name="Lock Length", default=False) 
    opt_lock_tilt: BoolProperty(name="Lock Tilt", default=False)
    opt_lock_radius: BoolProperty(name="Lock Radius", default=False)
    regular: BoolProperty(name="Regular", default=True, description="Distribute points evenly")
    interpolation: EnumProperty(
        name="Interpolation",
        items=(("cubic", "Cubic", "Natural cubic spline"),
               ("linear", "Linear", "Simple linear interpolation")),
        default='cubic'
    )
    iterations: IntProperty(name="Iterations", default=1, min=1, max=50)

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for spline in obj.data.splines:
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                num_points = len(points)
                if num_points < 3: continue
                sel_mask = [p.select_control_point if spline.type == 'BEZIER' else p.select for p in points]
                if not any(sel_mask): continue
                segments = []
                if all(sel_mask): segments.append(list(range(num_points)))
                else:
                    curr = []
                    for i, s in enumerate(sel_mask):
                        if s: curr.append(i)
                        else:
                            if curr: segments.append(curr); curr = []
                    if curr:
                        if spline.use_cyclic_u and sel_mask[0] and segments and segments[0][0] == 0:
                            segments[0] = curr + segments[0]
                        else: segments.append(curr)
                for seg in segments:
                    if len(seg) < 3: continue 
                    attrs = []
                    if self.relax_position: attrs.append('position')
                    if self.relax_tilt: attrs.append('tilt')
                    if self.relax_radius: attrs.append('radius')
                    if not attrs: continue
                    for _ in range(self.iterations):
                        pts_co = [points[i].co.to_3d() for i in seg]
                        seg_data = []
                        for i in seg:
                            d, p = [], points[i]
                            if self.relax_position: d.extend([p.co.x, p.co.y, p.co.z])
                            if self.relax_tilt: d.append(p.tilt)
                            if self.relax_radius: d.append(p.radius)
                            seg_data.append(d)
                        circ = (spline.use_cyclic_u and len(seg) == num_points)
                        ki, pi = relax_calculate_knots(len(seg_data), circ)
                        tk, tp = relax_calculate_t(pts_co, ki, pi, self.regular)
                        spls = []
                        for kp in range(len(ki)):
                            kd, t = [seg_data[k] for k in ki[kp]], tk[kp]
                            s = calculate_cubic_splines(t, kd) if self.interpolation == 'cubic' else calculate_linear_splines(t, kd)
                            spls.append(s)
                        moves = relax_calculate_verts(self.interpolation, tk, ki, tp, pi, spls)
                        for l_idx, n_vals in moves:
                            p = points[seg[l_idx]]; offset = 0
                            if self.relax_position:
                                o_co = p.co.to_3d(); n_co = (o_co + mathutils.Vector(n_vals[0:3])) / 2
                                if spline.type == 'BEZIER':
                                    delta = n_co - o_co; p.handle_left += delta; p.handle_right += delta; p.co = n_co
                                else:
                                    w = p.co[3]; p.co = n_co.to_4d(); p.co[3] = w
                                offset += 3
                            if self.relax_tilt: p.tilt = (p.tilt + n_vals[offset]) / 2; offset += 1
                            if self.relax_radius: p.radius = (p.radius + n_vals[offset]) / 2; offset += 1
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_space(Operator):
    bl_idname = "looptools_plus.curve_space"
    bl_label = "Space"
    bl_description = "Space points evenly along the curve"
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(name="Interpolation", items=(("cubic", "Cubic", ""), ("linear", "Linear", "")), default='cubic')
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')
    lock_x: BoolProperty(name="Lock X", default=False); lock_y: BoolProperty(name="Lock Y", default=False); lock_z: BoolProperty(name="Lock Z", default=False)

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'CURVE': continue
            for spline in obj.data.splines:
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                num_points = len(points)
                if num_points < 3: continue
                sel_mask = [p.select_control_point if spline.type == 'BEZIER' else p.select for p in points]
                if not any(sel_mask): continue
                segments = []
                if all(sel_mask): segments.append(list(range(num_points)))
                else:
                    curr = []
                    for i, s in enumerate(sel_mask):
                        if s: curr.append(i)
                        else:
                            if curr: segments.append(curr); curr = []
                    if curr:
                        if spline.use_cyclic_u and sel_mask[0] and segments and segments[0][0] == 0:
                            segments[0] = curr + segments[0]
                        else: segments.append(curr)
                for seg in segments:
                    if len(seg) < 2: continue
                    pts_co = [points[i].co.to_3d() for i in seg]
                    seg_data = []
                    for i in seg:
                        p = points[i]; seg_data.append([p.co.x, p.co.y, p.co.z, p.tilt, p.radius])
                    tk, tp = space_calculate_t(pts_co)
                    spls = calculate_cubic_splines(tk, seg_data) if self.interpolation == 'cubic' else calculate_linear_splines(tk, seg_data)
                    moves = space_calculate_verts(self.interpolation, tk, tp, spls)
                    infl = self.influence / 100.0
                    for l_idx, n_vals in moves:
                        p = points[seg[l_idx]]
                        o_co = p.co.to_3d(); t_co = mathutils.Vector(n_vals[0:3])
                        if self.lock_x: t_co.x = o_co.x
                        if self.lock_y: t_co.y = o_co.y
                        if self.lock_z: t_co.z = o_co.z
                        f_co = o_co.lerp(t_co, infl)
                        if spline.type == 'BEZIER':
                            delta = f_co - o_co; p.handle_left += delta; p.handle_right += delta; p.co = f_co
                        else:
                            w = p.co[3]; p.co = f_co.to_4d(); p.co[3] = w
                        p.tilt = p.tilt + (n_vals[3] - p.tilt) * infl
                        p.radius = p.radius + (n_vals[4] - p.radius) * infl
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_flatten(Operator):
    bl_idname = "looptools_plus.curve_flatten"
    bl_label = "Flatten"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        self.report({'INFO'}, "Flatten Curve not fully implemented yet"); return {'FINISHED'}

class LOOPTOOLSPLUS_OT_curve_circle(Operator):
    bl_idname = "looptools_plus.curve_circle"
    bl_label = "Circle"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        self.report({'INFO'}, "Circle Curve not fully implemented yet"); return {'FINISHED'}

# ########################################
# ##### UV Operators #####################
# ########################################

class UVLoopToolsBase:
    def get_uv_paths(self, bm, uv_layer):
        """
        Groups selected BMLoops into ordered paths (chains or cycles).
        A 'UV vertex' is a group of loops at the same mesh-vert that share the same UV coordinate.
        """
        # 1. Identify all selected loops and group them by (mesh_vert, uv_coord)
        # We use a small epsilon for UV coordinate matching
        def uv_key(uv):
            return (round(uv.x, 6), round(uv.y, 6))

        uv_nodes = {} # (vert, uv_key) -> [loops]
        for face in bm.faces:
            for l in face.loops:
                if l[uv_layer].select:
                    key = (l.vert, uv_key(l[uv_layer].uv))
                    if key not in uv_nodes:
                        uv_nodes[key] = []
                    uv_nodes[key].append(l)

        if not uv_nodes:
            return []

        # 2. Build adjacency graph between UV nodes
        # Two nodes are adjacent if they share a mesh edge AND that edge is part of a selected UV edge
        adj = {key: set() for key in uv_nodes}
        for face in bm.faces:
            for l in face.loops:
                l_next = l.link_loop_next
                key_curr = (l.vert, uv_key(l[uv_layer].uv))
                key_next = (l_next.vert, uv_key(l_next[uv_layer].uv))
                
                if key_curr in uv_nodes and key_next in uv_nodes:
                    # They might be connected!
                    adj[key_curr].add(key_next)
                    adj[key_next].add(key_curr)

        # 3. Traverse graph to find paths
        paths = []
        visited = set()
        
        # Keys sorted to ensure deterministic behavior
        all_keys = list(uv_nodes.keys())
        
        # Start with nodes that have only 1 neighbor (endpoints of chains)
        endpoints = [k for k in all_keys if len(adj[k]) == 1]
        for k in endpoints + [k for k in all_keys if k not in visited]:
            if k in visited:
                continue
            
            path = []
            curr = k
            while curr and curr not in visited:
                visited.add(curr)
                path.append(curr)
                # Find next neighbor not visited
                next_node = None
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        next_node = neighbor
                        break
                curr = next_node
            
            # Check for cycle if it's not a chain
            if path:
                first = path[0]
                last = path[-1]
                is_cyclic = first in adj[last] and len(path) > 2
                
                # Convert keys back to loop groups for processing
                paths.append({'nodes': [uv_nodes[node_key] for node_key in path], 'cyclic': is_cyclic})
                
        return paths

class LOOPTOOLSPLUS_OT_uv_relax(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_relax"
    bl_label = "Relax (UV)"
    bl_description = "Relax selected UV vertices"
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(name="Interpolation", items=(("cubic", "Cubic", ""), ("linear", "Linear", "")), default='cubic')
    iterations: IntProperty(name="Iterations", default=1, min=1, max=50)
    regular: BoolProperty(name="Regular", default=True)

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        for path_data in uv_paths:
            uv_run_nodes = path_data['nodes']
            is_circular = path_data['cyclic']
            if len(uv_run_nodes) < 3: continue
            
            for _ in range(self.iterations):
                # We use the UV of the first loop in each node as representative
                pts_co = [node[0][uv_layer].uv.to_3d() for node in uv_run_nodes]
                seg_data = [[node[0][uv_layer].uv.x, node[0][uv_layer].uv.y] for node in uv_run_nodes]
                
                ki, pi = relax_calculate_knots(len(seg_data), is_circular)
                tk, tp = relax_calculate_t(pts_co, ki, pi, self.regular)
                spls = []
                for kp in range(len(ki)):
                    kd, t = [seg_data[k] for k in ki[kp]], tk[kp]
                    s = calculate_cubic_splines(t, kd) if self.interpolation == 'cubic' else calculate_linear_splines(t, kd)
                    spls.append(s)
                
                moves = relax_calculate_verts(self.interpolation, tk, ki, tp, pi, spls)
                for l_idx, n_vals in moves:
                    # Apply to ALL loops in this node
                    new_uv = mathutils.Vector(n_vals)
                    for l in uv_run_nodes[l_idx]:
                        l[uv_layer].uv = (l[uv_layer].uv + new_uv) / 2
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_uv_space(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_space"
    bl_label = "Space (UV)"
    bl_description = "Space selected UV vertices evenly"
    bl_options = {'REGISTER', 'UNDO'}
    interpolation: EnumProperty(name="Interpolation", items=(("cubic", "Cubic", ""), ("linear", "Linear", "")), default='cubic')
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        infl = self.influence / 100.0
        for path_data in uv_paths:
            uv_run_nodes = path_data['nodes']
            if len(uv_run_nodes) < 2: continue
            
            pts_co = [node[0][uv_layer].uv.to_3d() for node in uv_run_nodes]
            seg_data = [[node[0][uv_layer].uv.x, node[0][uv_layer].uv.y] for node in uv_run_nodes]
            
            tk, tp = space_calculate_t(pts_co)
            spls = calculate_cubic_splines(tk, seg_data) if self.interpolation == 'cubic' else calculate_linear_splines(tk, seg_data)
            moves = space_calculate_verts(self.interpolation, tk, tp, spls)
            
            for l_idx, n_vals in moves:
                target = mathutils.Vector(n_vals)
                for l in uv_run_nodes[l_idx]:
                    l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_uv_circle(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_circle"
    bl_label = "Circle (UV)"
    bl_options = {'REGISTER', 'UNDO'}
    
    regular: BoolProperty(name="Regular", default=True, description="Distribute points evenly")
    influence: FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        infl = self.influence / 100.0

        for path_data in uv_paths:
            nodes = path_data['nodes']
            cyclic = path_data['cyclic']
            if len(nodes) < 3: continue
            
            # 1. Average center
            center = mathutils.Vector((0.0, 0.0))
            for node in nodes:
                center += node[0][uv_layer].uv
            center /= len(nodes)
            
            # 2. Average radius
            radius = sum((node[0][uv_layer].uv - center).length for node in nodes) / len(nodes)
            if radius < 1e-7: continue
            
            if self.regular:
                # Calculate angles and unwrap them to find the "arc" or "circle"
                angles = []
                for node in nodes:
                    vec = node[0][uv_layer].uv - center
                    angles.append(math.atan2(vec.y, vec.x))
                
                # Unwrap
                for i in range(1, len(angles)):
                    while angles[i] - angles[i-1] > math.pi: angles[i] -= 2*math.pi
                    while angles[i] - angles[i-1] < -math.pi: angles[i] += 2*math.pi
                
                if cyclic:
                    # For full circle, we span 2*pi
                    start_angle = angles[0]
                    for i, node in enumerate(nodes):
                        angle = start_angle + i * (2 * math.pi / len(nodes))
                        target = center + mathutils.Vector((math.cos(angle), math.sin(angle))) * radius
                        for l in node:
                            l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
                else:
                    # For arc, we span from first to last angle
                    start_angle = angles[0]
                    end_angle = angles[-1]
                    for i, node in enumerate(nodes):
                        angle = start_angle + (end_angle - start_angle) * (i / (len(nodes) - 1))
                        target = center + mathutils.Vector((math.cos(angle), math.sin(angle))) * radius
                        for l in node:
                            l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
            else:
                # Simply project radially
                for node in nodes:
                    for l in node:
                        vec = (l[uv_layer].uv - center)
                        if vec.length > 1e-7:
                            target = center + vec.normalized() * radius
                            l[uv_layer].uv = l[uv_layer].uv.lerp(target, infl)
                            
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

class LOOPTOOLSPLUS_OT_uv_flatten(Operator, UVLoopToolsBase):
    bl_idname = "looptools_plus.uv_flatten"
    bl_label = "Flatten (UV)"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH' or obj.mode != 'EDIT': return {'CANCELLED'}
        bm = bmesh.from_edit_mesh(obj.data)
        uv_layer = bm.loops.layers.uv.active
        if not uv_layer: return {'CANCELLED'}

        uv_paths = self.get_uv_paths(bm, uv_layer)
        if not uv_paths: return {'CANCELLED'}

        for path_data in uv_paths:
            uv_run_nodes = path_data['nodes']
            if len(uv_run_nodes) < 2: continue
            
            p1 = uv_run_nodes[0][0][uv_layer].uv
            p2 = uv_run_nodes[-1][0][uv_layer].uv
            line = p2 - p1
            if line.length > 0:
                line_norm = line.normalized()
                for node in uv_run_nodes:
                    for l in node:
                        rel = l[uv_layer].uv - p1
                        l[uv_layer].uv = p1 + line_norm * rel.dot(line_norm)
        bmesh.update_edit_mesh(obj.data)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_relax)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_space)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_flatten)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_curve_circle)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_relax)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_space)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_circle)
    bpy.utils.register_class(LOOPTOOLSPLUS_OT_uv_flatten)

def unregister():
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_flatten)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_circle)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_space)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_uv_relax)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_circle)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_flatten)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_space)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_OT_curve_relax)
