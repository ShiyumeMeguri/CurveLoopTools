import bpy
import mathutils
from . import utils

class CurveLoopTools_OT_flatten(bpy.types.Operator):
    bl_idname = "curve_looptools.flatten"
    bl_label = "Flatten"
    bl_description = "Flatten selected control points"
    bl_options = {'REGISTER', 'UNDO'}
    
    plane: bpy.props.EnumProperty(
        name="Plane",
        items=[('BEST_FIT', "Best Fit", "Calculate best fit plane"),
               ('NORMAL', "Normal", "Use average normal"),
               ('VIEW', "View", "Align to view")],
        default='BEST_FIT'
    )

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'CURVE' and context.mode == 'EDIT_CURVE'

    def execute(self, context):
        points_data = list(utils.get_selected_points(context))
        if not points_data:
            self.report({'WARNING'}, "No points selected")
            return {'CANCELLED'}

        # We care about coordinates
        # For Bezier, we flatten control points. Handles?
        # If we flatten control points, handles should probably be projected to the plane too 
        # to ensure the curve segment stays flat.
        
        # Collect positions
        positions = [p[2].co.copy() for p in points_data]
        
        # Calculate Plane
        co_list = [p[2].co.to_3d() for p in points_data]
        if self.plane == 'BEST_FIT':
            com, normal = utils.calculate_plane(co_list, method='best_fit')
        elif self.plane == 'NORMAL':
            # View normal or average normal?
            # 'NORMAL' in LoopTools often means Average Normal.
            # But calculating average normal for arbitrary points is tricky.
            # Use Best Fit for now as fallback.
            com, normal = utils.calculate_plane(co_list, method='best_fit')
        elif self.plane == 'VIEW':
            # Get View Normal from context
            view_mat = context.space_data.region_3d.view_matrix
            normal = view_mat.to_3x3().inverted() @ mathutils.Vector((0,0,1))
            com = sum(co_list, mathutils.Vector()) / len(co_list)

        
        # Project
        # P_proj = P - ((P - Center) dot Normal) * Normal
        
        for i, (spline, idx, bp) in enumerate(points_data):
            vec = bp.co - com
            dist = vec.dot(normal)
            proj = bp.co - (dist * normal)
            
            if spline.type == 'BEZIER':
                # Move Handles too!
                # Project handles relative to the projected point?
                # Or just project them absolutely?
                # Absolute projection ensures flatness.
                
                h1_vec = bp.handle_left - com
                h1_dist = h1_vec.dot(normal)
                h1_proj = bp.handle_left - (h1_dist * normal)
                
                h2_vec = bp.handle_right - com
                h2_dist = h2_vec.dot(normal)
                h2_proj = bp.handle_right - (h2_dist * normal)
                
                bp.co = proj
                bp.handle_left = h1_proj
                bp.handle_right = h2_proj
            else:
                # NURBS/POLY has 'co' as 4D (x,y,z,w) usually or 3D?
                # spline.points[i].co is Vector(4) for NURBS (x,y,z,w).
                
                if len(bp.co) == 4:
                    # preserve W
                    w = bp.co.w
                    bp.co = mathutils.Vector((proj.x, proj.y, proj.z, w))
                else:
                    bp.co = proj


        return {'FINISHED'}

class CurveLoopTools_OT_space(bpy.types.Operator):
    bl_idname = "curve_looptools.space"
    bl_label = "Space"
    bl_description = "Space selected control points evenly"
    bl_options = {'REGISTER', 'UNDO'}
    
    influence: bpy.props.FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'CURVE' and context.mode == 'EDIT_CURVE'

    def execute(self, context):
        points_data = list(utils.get_selected_points(context))
        if not points_data:
            return {'CANCELLED'}
        
        segments = utils.get_contiguous_segments(points_data)

        for seg in segments:
            if len(seg) < 3:
                 continue
            
            # Get points
            s = seg[0][0]
            if s.type == 'BEZIER':
                pts = [item[2] for item in seg]
                coords = [p.co.copy() for p in pts]
            else:
                pts = [item[2] for item in seg]
                coords = [p.co.to_3d() for p in pts]
            
            # Calculate total chord length
            total_len = 0.0
            lengths = [0.0]
            for k in range(len(coords)-1):
                d = (coords[k+1] - coords[k]).length
                total_len += d
                lengths.append(total_len)
                
            # Target average interval
            step = total_len / (len(coords) - 1)
            
            # New positions
            for k in range(1, len(coords)-1):
                 target_d = k * step
                 
                 # Find j where lengths[j] <= target_d < lengths[j+1]
                 for j in range(len(lengths)-1):
                     if lengths[j] <= target_d <= lengths[j+1]:
                         segment_len = lengths[j+1] - lengths[j]
                         if segment_len < 1e-6:
                             new_pos = coords[j]
                         else:
                             factor = (target_d - lengths[j]) / segment_len
                             new_pos = coords[j].lerp(coords[j+1], factor)
                         
                         # Apply influence
                         current_pos = coords[k]
                         final_pos = current_pos.lerp(new_pos, self.influence / 100.0)
                         
                         utils.move_bezier_point(pts[k], final_pos) if s.type == 'BEZIER' else setattr(pts[k], 'co', final_pos.to_4d() if len(pts[k].co)==4 else final_pos)
                         break
    
        return {'FINISHED'}

class CurveLoopTools_OT_circle(bpy.types.Operator):
    bl_idname = "curve_looptools.circle"
    bl_label = "Circle"
    bl_description = "Flatten to circle"
    bl_options = {'REGISTER', 'UNDO'}
    
    fit: bpy.props.EnumProperty(
        name="Fit",
        items=[('BEST_FIT', "Best Fit", "Calculate best fit circle"),
               ('INSIDE', "Inside", "Fit inside points"),
               ('OUTSIDE', "Outside", "Fit outside points")],
        default='BEST_FIT'
    )
    flatten: bpy.props.BoolProperty(name="Flatten", default=True, description="Flatten points to plane")
    radius: bpy.props.FloatProperty(name="Radius", default=0.0, min=0.0, description="Custom radius (0 = auto)")
    regular: bpy.props.BoolProperty(name="Regular", default=True, description="Distribute points evenly")
    influence: bpy.props.FloatProperty(name="Influence", default=100.0, min=0.0, max=100.0, subtype='PERCENTAGE')

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'CURVE' and context.mode == 'EDIT_CURVE'

    def execute(self, context):
        points_data = list(utils.get_selected_points(context))
        if not points_data:
            return {'CANCELLED'}
        
        # 1. Calc Plane
        co_list = [p[2].co.to_3d() for p in points_data]
        com, normal = utils.calculate_plane(co_list, method='best_fit')
        
        if not self.flatten:
             # If not flatten, we project to the 'cylinder' defined by the circle?
             # LoopTools Circle without flatten projects points to the circle cylinder along normal?
             # Or just projects to closest point on circle ring in 3D? usually on plane.
             # Standard LoopTools 'Circle' flattens by default.
             pass
             
        # 2. Calc Radius
        # Project points to plane to get 2D coords relative to COM
        # Base Axis X, Y
        z_axis = normal
        x_axis = mathutils.Vector((1,0,0))
        if abs(z_axis.x) > 0.9: x_axis = mathutils.Vector((0,1,0))
        y_axis = z_axis.cross(x_axis).normalized()
        x_axis = y_axis.cross(z_axis).normalized()
        
        avg_dist = 0.0
        projected_pts = []
        for p in co_list:
            vec = p - com
            # project to plane
            dist_n = vec.dot(z_axis)
            p_on_plane = p - dist_n * z_axis
            
            # Distance to center (radius estimate)
            r_vec = p_on_plane - com
            r = r_vec.length
            avg_dist += r
            projected_pts.append({'pt': p, 'on_plane': p_on_plane, 'r': r, 'vec': r_vec})
            
        avg_dist /= len(co_list)
        radius = self.radius if self.radius > 0 else avg_dist
        
        # 3. Calculate new positions
        new_positions = []
        
        if self.regular:
             # Sort by angle
             # Angle relative to X axis
             for item in projected_pts:
                 v = item['vec']
                 x = v.dot(x_axis)
                 y = v.dot(y_axis)
                 item['angle'] = math.atan2(y, x)
                 
             projected_pts.sort(key=lambda x: x['angle'])
             
             # Distribute evenly
             step = 2 * math.pi / len(projected_pts)
             # Start angle? align to first point?
             # Keep total rotation minimal? 
             # Align first point's angle
             start_angle = projected_pts[0]['angle'] 
             
             for i, item in enumerate(projected_pts):
                 theta = start_angle + i * step
                 # circle pos
                 circle_pos = com + radius * (math.cos(theta) * x_axis + math.sin(theta) * y_axis)
                 
                 if not self.flatten:
                     # Add back original height?
                     vec = item['pt'] - com
                     height = vec.dot(z_axis)
                     circle_pos += height * z_axis
                     
                 new_positions.append(circle_pos)
                 
             # We need to map back to original indices.
             # But we sorted 'projected_pts'.
             # We need to preserve the mapping to 'points_data'.
             # Wait, selection order in 'points_data' might not be loop order.
             # If points are connected, we should respect that order.
             # 'get_selected_points' yields by spline index.
             # If 'Regular', we usually want to follow the spline connectivity.
             
             # Re-approach: Iterate segments.
             pass
        
        # Simple implementation without 'Regular' sorting for now if not segments
        # If Regular, we must respect Spline order.
        
        segments = utils.get_contiguous_segments(points_data)
        # Flatten segments list
        # points_data is already sorted by spline/index in get_selected_points if we trusted utils?
        # utils.get_selected_points iterates splines/points in order.
        
        # For Circle, usually we act on a closed loop.
        # Construct the "Loop" from segments.
        
        # If Regular:
        if self.regular:
            # Re-collect all points in order
            ordered_items = []
            for seg in segments:
                ordered_items.extend(seg)
            
            # calculate angle of FIRST point relative to center to set phase?
            # Or fit line best?
            
            # Let's assume the points are in correct order along the perimeter.
            total_pts = len(ordered_items)
            step = 2 * math.pi / total_pts
            
            # We need a starting angle.
            # Use the angle of the first point.
            p0 = ordered_items[0][2].co.to_3d()
            v0 = p0 - com
            # Project v0 to plane
            v0_plane = v0 - v0.dot(z_axis) * z_axis
            start_angle = math.atan2(v0_plane.dot(y_axis), v0_plane.dot(x_axis))
            
            for i, (s, idx, bp) in enumerate(ordered_items):
                theta = start_angle + i * step
                # Target pos
                target = com + radius * (math.cos(theta) * x_axis + math.sin(theta) * y_axis)
                
                if not self.flatten:
                    # Height
                    h = (bp.co.to_3d() - com).dot(z_axis)
                    target += h * z_axis
                
                factor = self.influence / 100.0
                current = bp.co.to_3d()
                final = current.lerp(target, factor)
                utils.move_bezier_point(bp, final) if s.type == 'BEZIER' else setattr(bp, 'co', final.to_4d() if len(bp.co)==4 else final)

        else:
            # Irregular (Project nearest)
            for s, idx, bp in points_data:
                p = bp.co.to_3d()
                vec = p - com
                h = vec.dot(z_axis)
                p_plane = p - h * z_axis
                
                if p_plane.length_squared < 1e-6:
                    dir_vec = mathutils.Vector((1,0,0))
                else:
                    dir_vec = p_plane.normalized()
                
                target = com + dir_vec * radius
                
                if not self.flatten:
                    target += h * z_axis
                
                factor = self.influence / 100.0
                current = bp.co.to_3d()
                final = current.lerp(target, factor)
                
                utils.move_bezier_point(bp, final) if s.type == 'BEZIER' else setattr(bp, 'co', final.to_4d() if len(bp.co)==4 else final)
                
        return {'FINISHED'}

class CurveLoopTools_OT_relax(bpy.types.Operator):
    bl_idname = "curve_looptools.relax"
    bl_label = "Relax"
    bl_description = "Relax selected points"
    bl_options = {'REGISTER', 'UNDO'}
    
    iterations: bpy.props.IntProperty(name="Iterations", default=1, min=1, max=25)
    regular: bpy.props.BoolProperty(name="Regular", default=True)
    
    relax_position: bpy.props.BoolProperty(name="Position", default=True)
    relax_tilt: bpy.props.BoolProperty(name="Tilt", default=False)
    relax_radius: bpy.props.BoolProperty(name="Radius", default=False)
    lock_radius: bpy.props.BoolProperty(name="Lock Radius", default=False, description="Preserve original radius volume")
    lock_tilt: bpy.props.BoolProperty(name="Lock Tilt", default=False, description="Preserve original total tilt")
    lock_length: bpy.props.BoolProperty(name="Lock Length", default=False, description="Preserve original curve length")

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'CURVE' and context.mode == 'EDIT_CURVE'

    def execute(self, context):
        points_data = list(utils.get_selected_points(context))
        segments = utils.get_contiguous_segments(points_data)
        
        for _ in range(self.iterations):
            # Calculate new positions/attributes for all segments
            updates = []
            
            # Tracking for Locks
            affected_radius_bp = []
            initial_radius_sum = 0.0
            
            affected_tilt_bp = []
            initial_tilt_sum = 0.0
            
            # For length, we need to track per segment because they are independent chains
            # Map segment_index -> { 'initial_len', 'points_bp' }
            seg_length_data = {} 

            # 1. Pre-calculate Sums/Lengths
            for s_idx, seg in enumerate(segments):
                # Lock Radius
                if self.relax_radius and self.lock_radius:
                    for i, (spline, idx, bp) in enumerate(seg):
                        affected_radius_bp.append(bp)
                        initial_radius_sum += bp.radius
                
                # Lock Tilt
                if self.relax_tilt and self.lock_tilt:
                    for i, (spline, idx, bp) in enumerate(seg):
                         if hasattr(bp, 'tilt'):
                            affected_tilt_bp.append(bp)
                            initial_tilt_sum += bp.tilt
                            
                # Lock Length
                if self.relax_position and self.lock_length:
                    # Calculate total chord length of this segment
                    current_len = 0.0
                    pts = [item[2] for item in seg]
                    # We need ordered coordinates.
                    # 'seg' is ordered list from contiguous segments function
                    coords = [p.co.to_3d() for p in pts]
                    for k in range(len(coords)-1):
                        current_len += (coords[k+1] - coords[k]).length
                    
                    seg_length_data[s_idx] = {'initial': current_len, 'pts': pts}

            # 2. Calculate Updates (Relaxation)
            for seg in segments:
                s = seg[0][0]
                total_pts = len(s.bezier_points) if s.type == 'BEZIER' else len(s.points)
                
                for i, (spline, idx, bp) in enumerate(seg):
                    prev_idx = idx - 1
                    next_idx = idx + 1
                    
                    if spline.use_cyclic_u:
                        prev_idx %= total_pts
                        next_idx %= total_pts
                    else:
                        if prev_idx < 0: prev_idx = 0 
                        if next_idx >= total_pts: next_idx = total_pts - 1
                    
                    if (prev_idx == idx or next_idx == idx) and not spline.use_cyclic_u:
                        continue

                    update_data = {}
                    
                    # Position
                    if self.relax_position:
                        if spline.type == 'BEZIER':
                            p_prev = spline.bezier_points[prev_idx].co.to_3d()
                            p_next = spline.bezier_points[next_idx].co.to_3d()
                        else:
                            p_prev = spline.points[prev_idx].co.to_3d()
                            p_next = spline.points[next_idx].co.to_3d()
                        
                        target_pos = (p_prev + p_next) / 2
                        update_data['pos'] = target_pos

                    # Tilt
                    if self.relax_tilt:
                        if spline.type == 'BEZIER':
                            t_prev = spline.bezier_points[prev_idx].tilt
                            t_next = spline.bezier_points[next_idx].tilt
                        else:
                            t_prev = spline.points[prev_idx].tilt
                            t_next = spline.points[next_idx].tilt
                        target_tilt = (t_prev + t_next) / 2
                        update_data['tilt'] = target_tilt

                    # Radius
                    if self.relax_radius:
                        if spline.type == 'BEZIER':
                            r_prev = spline.bezier_points[prev_idx].radius
                            r_next = spline.bezier_points[next_idx].radius
                        else:
                            r_prev = spline.points[prev_idx].radius
                            r_next = spline.points[next_idx].radius
                        target_radius = (r_prev + r_next) / 2
                        update_data['radius'] = target_radius
                    
                    if update_data:
                        updates.append((bp, update_data, spline))

            # 3. Apply Updates
            for bp, data, spline in updates:
                if 'pos' in data:
                    target = data['pos']
                    current = bp.co.to_3d()
                    new_pos = current.lerp(target, 0.5)
                    if isinstance(bp, bpy.types.BezierSplinePoint):
                        utils.move_bezier_point(bp, new_pos)
                    else:
                        if len(bp.co)==4: bp.co = new_pos.to_4d()
                        else: bp.co = new_pos
                
                if 'tilt' in data and hasattr(bp, 'tilt'):
                    bp.tilt = bp.tilt * 0.5 + data['tilt'] * 0.5
                    
                if 'radius' in data:
                    bp.radius = bp.radius * 0.5 + data['radius'] * 0.5
            
            # 4. Restore Locks
            
            # Lock Radius
            if self.relax_radius and self.lock_radius and affected_radius_bp and initial_radius_sum > 0:
                new_sum = sum([bp.radius for bp in affected_radius_bp])
                if new_sum > 1e-6:
                    factor = initial_radius_sum / new_sum
                    for bp in affected_radius_bp:
                        bp.radius *= factor
            
            # Lock Tilt
            if self.relax_tilt and self.lock_tilt and affected_tilt_bp and initial_tilt_sum != 0:
                 # Tilt can be negative or zero sum? 
                 # If sum is 0, we can't scale.
                 # Usually tilt is around 0? No, usually 0 is flat.
                 # Let's try simple scaling. If sum is tiny, skip to avoid explosion.
                 new_sum = sum([bp.tilt for bp in affected_tilt_bp])
                 if abs(new_sum) > 1e-6:
                     factor = initial_tilt_sum / new_sum
                     for bp in affected_tilt_bp:
                         bp.tilt *= factor
                         
            # Lock Length (Separately per segment)
            if self.relax_position and self.lock_length:
                for s_idx, info in seg_length_data.items():
                    target_len = info['initial']
                    pts = info['pts']
                    if len(pts) < 2 or target_len < 1e-6: continue
                    
                    # Current smoothed coords
                    coords = [p.co.to_3d() for p in pts]
                    new_len = 0.0
                    for k in range(len(coords)-1):
                        new_len += (coords[k+1] - coords[k]).length
                    
                    if new_len > 1e-6:
                        factor = target_len / new_len
                        # Scale segments?
                        # Simple approach: Rebuild chain from start point (pts[0])
                        # P_next = P_curr + (Dir * (old_len * factor)) ??
                        # Wait, (old_len * factor) is just new_len * factor = target_len ... no
                        # We want the DISTANCE between points to scale up.
                        
                        # BUT, simply scaling vector (P_i+1 - P_i) by factor restores total length
                        # but preserves the new smoothed direction!
                        
                        start_co = coords[0]
                        current_trace = start_co
                        
                        # We apply from start to end (propagating)
                        # Caution: if it's cyclic, this might break the loop closure?
                        # 'get_contiguous_segments' handles cyclic by potentially returning full loop (duplicate start/end?)
                        # If duplicate start/end, we should handle carefully.
                        # Assuming linear chain for now.
                        
                        for k in range(len(coords)-1):
                            vec = coords[k+1] - coords[k]
                            vec_len = vec.length
                            # Scale vector
                            new_vec = vec * factor
                            
                            # Next point
                            next_pos = current_trace + new_vec
                            
                            # Apply to pts[k+1]
                            # pts[0] stays fixed.
                            
                            bp = pts[k+1]
                            if isinstance(bp, bpy.types.BezierSplinePoint):
                                utils.move_bezier_point(bp, next_pos)
                            else:
                                if len(bp.co)==4: bp.co = next_pos.to_4d()
                                else: bp.co = next_pos
                                
                            current_trace = next_pos

        return {'FINISHED'}

classes = (
    CurveLoopTools_OT_flatten,
    CurveLoopTools_OT_space,
    CurveLoopTools_OT_circle,
    CurveLoopTools_OT_relax,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
