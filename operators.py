import bpy
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
        
        q = [0.0] # Placeholder, will be ignored/overwritten loop starts at 1
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
            # result[i] will contain lists of [a,b,c,d,x] for each dimension
            result[i].append([a[i], b[i], c[i], d[i], x[i]])

    # Reformat result to match LoopTools structure: list of splines per segment
    # LoopTools returns: list of [ [ax,bx...], [ay,by...], [az,bz...] ] per segment
    splines = []
    for i in range(n - 1):
        # result[i] is [[a,b,c,d,x]_dim0, [a,b,c,d,x]_dim1, ...]
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
        deconversion = lambda x: x[0]
    else:
        dim = len(knots[0])
        conversion = lambda x: list(x)
        deconversion = lambda x: x
        
    for i in range(len(knots) - 1):
        a = conversion(knots[i])
        b = conversion(knots[i + 1])
        # d = b - a
        d = [b[k] - a[k] for k in range(dim)]
        
        t = tknots[i]
        u = tknots[i + 1] - t
        
        # Structure: [ [a0, d0, t, u], [a1, d1, t, u], ... ]
        segment_splines = []
        for k in range(dim):
            segment_splines.append([a[k], d[k], t, u])
        splines.append(segment_splines)
        
    return splines


# ########################################
# ##### Relax functions for Curves #######
# ########################################

def relax_calculate_knots(points_len, circular):
    """
    Splits points into two sets (Even/Odd) for smoothing, similar to LoopTools.
    points_len: number of points in the loop
    circular: boolean
    """
    knots = [[], []]
    points = [[], []]
    
    loop = list(range(points_len)) # indices
    
    if circular:
        if len(loop) % 2 == 1: # odd
             extend = [False, True, 0, 1, 0, 1]
        else: # even
             extend = [True, False, 0, 1, 1, 2]
    else:
        if len(loop) % 2 == 1: # odd
             extend = [False, False, 0, 1, 1, 2]
        else: # even
             extend = [False, False, 0, 1, 1, 2]
             
    for j in range(2):
        temp_loop = loop[:]
        if extend[j]:
            temp_loop = [loop[-1]] + loop + [loop[0]]
            
        # Knots indices (every 2nd point)
        k_indices = []
        start_k = extend[2 + 2 * j]
        for i in range(start_k, len(temp_loop), 2):
            k_indices.append(temp_loop[i])
        knots[j] = k_indices
        
        # Points to move indices
        p_indices = []
        start_p = extend[3 + 2 * j]
        for i in range(start_p, len(temp_loop), 2):
            idx = temp_loop[i]
            if idx == loop[-1] and not circular:
               continue
            # Avoid duplicates if logic produces them (LoopTools check)
            if len(p_indices) == 0:
                p_indices.append(idx)
            elif idx != p_indices[0]:
                p_indices.append(idx)
        points[j] = p_indices
        
        if circular:
            if knots[j][0] != knots[j][-1]:
                knots[j].append(knots[j][0])
                
    if len(points[1]) == 0:
        knots.pop(1)
        points.pop(1)
        
    return knots, points

def relax_calculate_t(points_co, knots, points_indices, regular):
    """
    Calculates t parameter (length along curve) for knots and points.
    points_co: list of Vectors (positions) corresponding to segment indices
    """
    all_tknots = []
    all_tpoints = []
    
    for i in range(len(knots)):
        k_list = knots[i]
        p_list = points_indices[i]
        
        mix = []
        
        nk = len(k_list)
        np = len(p_list)
        
        # Safe interleaving
        max_len = max(nk, np)
        for j in range(max_len):
            if j < nk:
                mix.append((True, k_list[j]))
            if j < np:
                mix.append((False, p_list[j]))
                
        # Now calculate lengths
        len_total = 0
        loc_prev = None
        tknots = []
        tpoints = []
        
        for is_knot, idx in mix:
            # get location from points_co
            loc = points_co[idx]
                
            if loc_prev is None:
                loc_prev = loc
            
            dist = (loc - loc_prev).length
            len_total += dist
            
            if is_knot:
                tknots.append(len_total)
            else:
                tpoints.append(len_total)
                
            loc_prev = loc
            
        if regular:
            # Distribute points evenly between knots
            # LoopTools: tpoints[p] = (tknots[p] + tknots[p+1]) / 2
            # This places the point exactly in the middle (parametrically) of the two knots.
            new_tpoints = []
            for p_idx in range(len(tpoints)):
                # Ensure we have enough knots. 
                # With 'mix' order K, P, K, P... 
                # point[p] between knot[p] and knot[p+1] ?
                # The assumption is that tknots has indices corresponding to the same segments...
                # Note: LoopTools assumption: mix is K0, P0, K1, P1 ...
                # tknots[p] is K0 length, tknots[p+1] is K1 length.
                # So P0 should be at (K0+K1)/2. 
                
                # Verify indices:
                # If j=0: mix has K0, P0.
                # tknots will have 1 element, tpoints 1 element.
                # next iteration: K1, P1...
                # So indices align.
                
                if p_idx + 1 < len(tknots):
                    val = (tknots[p_idx] + tknots[p_idx+1]) / 2.0
                    new_tpoints.append(val)
                else:
                    # Fallback if at end
                     # If circular and wrap occurred?
                    if len(knots) > 1 and len(knots[0]) > 0: # just safety
                         new_tpoints.append(tpoints[p_idx])
                    else:
                         new_tpoints.append(tpoints[p_idx])

            tpoints = new_tpoints
            
        all_tknots.append(tknots)
        all_tpoints.append(tpoints)
        
    return all_tknots, all_tpoints

def get_data_from_index(points, index, attributes):
    # Retrieve data vector [x, y, z, tilt, radius]
    p = points[index]
    data = []
    # Position
    if 'position' in attributes:
        data.extend([p.co.x, p.co.y, p.co.z])
    # Tilt
    if 'tilt' in attributes:
        data.append(p.tilt)
    # Radius
    if 'radius' in attributes:
        data.append(p.radius)
    return data

def relax_calculate_verts(interpolation, tknots, knots, tpoints, points_indices, splines):
    """
    Interpolates new values for 'points' based on splines.
    Returns list of tuples: (index, [new_values])
    """
    moves = []
    
    # Iterate over the two passes (or 1)
    for i in range(len(knots)): # i is pass index (0 or 1)
        k_list = knots[i]
        p_list = points_indices[i]
        tk = tknots[i] # parametric positions of knots
        tp = tpoints[i] # parametric positions of points (targets)
        seg_splines = splines[i] # list of splines
        
        for j, p_idx in enumerate(p_list):
            if j >= len(tp):
                continue
                
            m = tp[j] # target t
            
            # Find knot n where tk[n] <= m
            n = -1
            # fast check if m matches a knot exactly (rare)
            if m in tk:
                 n = tk.index(m)
            else:
                # Find interval
                for k_idx in range(len(tk)):
                    if tk[k_idx] > m:
                        n = k_idx - 1
                        break
                if n == -1:
                    n = len(tk) - 1 # Should not happen if m inside range
                    
            # Clamp n to valid spline segments
            if n > len(seg_splines) - 1:
                n = len(seg_splines) - 1
            if n < 0:
                n = 0
                
            # Evaluate Spline
            new_vals = []
            
            if interpolation == 'cubic':
                # seg_splines[n] is list of dims: [[a,b,c,d,x], [a,b,c,d,x]...]
                dims = seg_splines[n]
                for d_idx in range(len(dims)):
                    a, b, c, d_coeff, tx = dims[d_idx]
                    dt = m - tx
                    # The LoopTools code defines x (tx) as the starting t of the segment
                    # So dt is distance from start of segment.
                    val = a + b*dt + c*(dt**2) + d_coeff*(dt**3)
                    new_vals.append(val)
            else: # linear
                # seg_splines[n] is [[a, d, t, u], ...]
                dims = seg_splines[n]
                for d_idx in range(len(dims)):
                    a, d_val, t, u = dims[d_idx]
                    if u == 0: u = 1e-8
                    val = ((m - t) / u) * d_val + a
                    new_vals.append(val)
                    
            moves.append((p_idx, new_vals))
            
    return moves


class CurveRelax(Operator):
    bl_idname = "curve_looptools.relax"
    bl_label = "Relax"
    bl_description = "Relax the curve, smoothing it out"
    bl_options = {'REGISTER', 'UNDO'}

    relax_position: BoolProperty(name="Relax Position", default=True)
    relax_tilt: BoolProperty(name="Relax Tilt", default=False)
    relax_radius: BoolProperty(name="Relax Radius", default=False)
    
    # Locking - kept from UI, mainly to skip dimensions if needed
    opt_lock_length: BoolProperty(name="Lock Length", default=False) 
    opt_lock_tilt: BoolProperty(name="Lock Tilt", default=False) # Redundant with relax_tilt=False?
    opt_lock_radius: BoolProperty(name="Lock Radius", default=False)
    
    regular: BoolProperty(name="Regular", default=True, description="Distribute points evenly")
    
    interpolation: EnumProperty(
        name="Interpolation",
        items=(("cubic", "Cubic", "Natural cubic spline"),
               ("linear", "Linear", "Simple linear interpolation")),
        default='cubic'
    )
    iterations: IntProperty(name="Iterations", default=1, min=1, max=50) # Allow int input

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'CURVE':
                continue
                
            for spline in obj.data.splines:
                # Identify chunks of selected points
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                num_points = len(points)
                if num_points < 3:
                    continue

                if spline.type == 'BEZIER':
                    selection_mask = [p.select_control_point for p in points]
                else:
                    selection_mask = [p.select for p in points]
                if not any(selection_mask):
                    continue
                    
                # Find segments
                segments = []
                if all(selection_mask):
                    segments.append(list(range(num_points)))
                else:
                    # Find runs
                    current_run = []
                    for i, sel in enumerate(selection_mask):
                        if sel:
                            current_run.append(i)
                        else:
                            if current_run:
                                segments.append(current_run)
                                current_run = []
                    if current_run:
                        # Check cyclic wrap
                        if spline.use_cyclic_u and selection_mask[0]:
                            if len(segments) > 0 and segments[0][0] == 0:
                                # Merge last and first
                                segments[0] = current_run + segments[0]
                            else:
                                segments.append(current_run)
                        else:
                            segments.append(current_run)

                # Now Relax each segment
                for seg_indices in segments:
                    if len(seg_indices) < 3:
                        continue 
                        
                    # Attributes to relax
                    attrs = []
                    if self.relax_position: attrs.append('position')
                    if self.relax_tilt: attrs.append('tilt')
                    if self.relax_radius: attrs.append('radius')
                    
                    if not attrs:
                        continue

                    # Iterations
                    for _ in range(self.iterations):
                        # Extract 3D Positions for T calc (Always needed)
                        # We use seg_indices to map local 0..M to spline indices
                        segment_points_co = []
                        for idx in seg_indices:
                            segment_points_co.append(points[idx].co.to_3d())
                            
                        # Extract Data to Relax
                        segment_data = []
                        for idx in seg_indices:
                            segment_data.append(get_data_from_index(points, idx, attrs))
                            
                        # Is circular?
                        is_circular = (spline.use_cyclic_u and len(seg_indices) == num_points)
                        
                        # 1. Calculate Knots Indices
                        knots_indices, points_indices = relax_calculate_knots(len(segment_data), is_circular)
                        
                        # 2. Calculate t parameters (Using Positions!)
                        tknots, tpoints = relax_calculate_t(segment_points_co, knots_indices, points_indices, self.regular)
                        
                        # 3. Calculate Splines (Using Relax Data)
                        splines = []
                        for k_pass_idx in range(len(knots_indices)):
                            # Gather knot data (values)
                            k_data = [segment_data[k] for k in knots_indices[k_pass_idx]]
                            tk = tknots[k_pass_idx]
                            
                            if self.interpolation == 'cubic':
                                s = calculate_cubic_splines(tk, k_data)
                            else:
                                s = calculate_linear_splines(tk, k_data)
                            splines.append(s)
                            
                        # 4. Calculate New Verts
                        move_list = relax_calculate_verts(self.interpolation, tknots, knots_indices, tpoints, points_indices, splines)
                        
                        # Apply changes
                        for local_idx, new_val_list in move_list:
                            spline_idx = seg_indices[local_idx]
                            p = points[spline_idx]
                            
                            # Update attributes
                            offset = 0
                            if self.relax_position:
                                old_co_3d = p.co.to_3d()
                                new_co_3d = mathutils.Vector(new_val_list[offset:offset+3])
                                avg_co_3d = (old_co_3d + new_co_3d) / 2
                                
                                # Handle Handles
                                if spline.type == 'BEZIER':
                                    delta = avg_co_3d - old_co_3d
                                    p.handle_left += delta
                                    p.handle_right += delta
                                    p.co = avg_co_3d
                                else:
                                    # For NURBS/Poly, co is 4D (x,y,z,w)
                                    # Preserve w
                                    w = p.co[3]
                                    p.co = avg_co_3d.to_4d() # sets w to 1.0 implicitly
                                    p.co[3] = w # restore w
                                    
                                offset += 3
                                
                            if self.relax_tilt:
                                old_tilt = p.tilt
                                new_tilt = new_val_list[offset]
                                p.tilt = (old_tilt + new_tilt) / 2
                                offset += 1
                                
                            if self.relax_radius:
                                old_rad = p.radius
                                new_rad = new_val_list[offset]
                                p.radius = (old_rad + new_rad) / 2
                                offset += 1
                                
        return {'FINISHED'}


class CurveFlatten(Operator):
    bl_idname = "curve_looptools.flatten"
    bl_label = "Flatten"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        self.report({'INFO'}, "Flatten not implemented yet")
        return {'FINISHED'}

class CurveCircle(Operator):
    bl_idname = "curve_looptools.circle"
    bl_label = "Circle"
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        self.report({'INFO'}, "Circle not implemented yet")
        return {'FINISHED'}

class CurveSpace(Operator):
    bl_idname = "curve_looptools.space"
    bl_label = "Space"
    bl_description = "Space points evenly (Not Implemented yet)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.report({'INFO'}, "Space Operator not implemented for Curves yet.")
        return {'FINISHED'}

def register():
    bpy.utils.register_class(CurveRelax)
    bpy.utils.register_class(CurveSpace)
    bpy.utils.register_class(CurveFlatten)
    bpy.utils.register_class(CurveCircle)

def unregister():
    bpy.utils.unregister_class(CurveCircle)
    bpy.utils.unregister_class(CurveFlatten)
    bpy.utils.unregister_class(CurveSpace)
    bpy.utils.unregister_class(CurveRelax)
