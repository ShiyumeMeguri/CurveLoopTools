import bpy
import mathutils

def get_selected_points(context):
    """
    Returns a list of selected points from the active object's curves.
    Yields: (spline, index, point)
    """
    obj = context.active_object
    if not obj or obj.type != 'CURVE':
        return

    # Helper to check selection for all curve types
    for spline in obj.data.splines:
        if spline.type == 'BEZIER':
            for i, bp in enumerate(spline.bezier_points):
                if bp.select_control_point:
                    yield (spline, i, bp)
        elif spline.type == 'POLY' or spline.type == 'NURBS':
            for i, p in enumerate(spline.points):
                if p.select:
                    yield (spline, i, p)

def move_bezier_point(bp, new_co):
    """
    Moves a Bezier point to new_co and offsets handles to maintain shape.
    """
    offset = new_co - bp.co
    bp.co = new_co
    bp.handle_left += offset
    bp.handle_right += offset


# calculate the determinant of a matrix
def matrix_determinant(m):
    determinant = m[0][0] * m[1][1] * m[2][2] + m[0][1] * m[1][2] * m[2][0] \
        + m[0][2] * m[1][0] * m[2][1] - m[0][2] * m[1][1] * m[2][0] \
        - m[0][1] * m[1][0] * m[2][2] - m[0][0] * m[1][2] * m[2][1]

    return(determinant)


# custom matrix inversion, to provide higher precision than the built-in one
def matrix_invert(m):
    r = mathutils.Matrix((
        (m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2],
         m[0][1] * m[1][2] - m[0][2] * m[1][1]),
        (m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0],
         m[0][2] * m[1][0] - m[0][0] * m[1][2]),
        (m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1],
         m[0][0] * m[1][1] - m[0][1] * m[1][0])))

    return (r * (1 / matrix_determinant(m)))

def calculate_plane(points, method="best_fit"):
    """
    Calculates the best-fit plane for a list of coordinates.
    Returns: (center_of_mass, normal)
    """
    if not points:
        return mathutils.Vector(), mathutils.Vector((0,0,1))

    # Center of mass
    com = mathutils.Vector((0.0, 0.0, 0.0))
    for p in points:
        com += p
    com /= len(points)
    x, y, z = com

    if method == 'best_fit':
        # creating the covariance matrix
        mat = mathutils.Matrix(((0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0),
                                (0.0, 0.0, 0.0),
                                ))
        for loc in points:
            mat[0][0] += (loc[0] - x) ** 2
            mat[1][0] += (loc[0] - x) * (loc[1] - y)
            mat[2][0] += (loc[0] - x) * (loc[2] - z)
            mat[0][1] += (loc[1] - y) * (loc[0] - x)
            mat[1][1] += (loc[1] - y) ** 2
            mat[2][1] += (loc[1] - y) * (loc[2] - z)
            mat[0][2] += (loc[2] - z) * (loc[0] - x)
            mat[1][2] += (loc[2] - z) * (loc[1] - y)
            mat[2][2] += (loc[2] - z) ** 2

        # calculating the normal to the plane
        normal = False
        try:
            mat = matrix_invert(mat)
        except:
             # fallback for singular matrix
            normal = mathutils.Vector((0,0,1))
        
        if not normal:
             # iterative power method to find eigenvector with smallest eigenvalue (normal)
             # Actually LoopTools uses inverse iteration to find smallest eigenvalue?
             # If we inverted the covariance matrix, the LARGEST eigenvalue of the inverse 
             # corresponds to the SMALLEST eigenvalue of original (which is the variance in normal direction).
             # So we want the eigenvector for the LARGEST eigenvalue of 'mat' (which is inverted covariance).
             
            itermax = 500
            vec2 = mathutils.Vector((1.0, 1.0, 1.0))
            for i in range(itermax):
                vec = vec2
                vec2 = mat @ vec
                if vec2.length != 0:
                    vec2 /= vec2.length
                if (vec2 - vec).length < 1e-6:
                    break
            normal = vec2
            
    elif method == 'normal':
        # Average normal - not really defined for text/curves point list unless we have tangents?
        # For now, fallback to best fit
        return calculate_plane(points, method='best_fit')

    elif method == 'view':
        # handled by caller using context
        normal = mathutils.Vector((0,0,1))

    return com, normal

def get_point_resolution(spline):
    """Return resolution for spline or defaults."""
    # Bezier resolution?
    # Not used directly for point manipulation but maybe for sampling
    return 12

def get_contiguous_segments(points_data):
    """
    Groups selected points into contiguous segments based on spline index and point index.
    points_data: list of (spline, index, point)
    Returns: list of segments, where each segment is a list of (spline, index, point) sorted by index.
    """
    # Group by spline
    splines = {}
    for s, i, bp in points_data:
        sid = id(s)
        if sid not in splines:
            splines[sid] = {'spline': s, 'items': []}
        splines[sid]['items'].append((s, i, bp))
        
    segments = []
    
    for sid, data in splines.items():
        items = sorted(data['items'], key=lambda x: x[1])
        if not items: continue
        
        current_seg = [items[0]]
        for k in range(1, len(items)):
            prev = items[k-1][1]
            curr = items[k][1]
            if curr == prev + 1:
                current_seg.append(items[k])
            else:
                segments.append(current_seg)
                current_seg = [items[k]]
        segments.append(current_seg)
        
        # Handle Cyclic
        s = data['spline']
        use_cyclic = s.use_cyclic_u
        num_points = len(s.bezier_points) if s.type == 'BEZIER' else len(s.points)
        
        if use_cyclic and len(segments) > 1:
             first = segments[0]
             last = segments[-1]
             if first[0][1] == 0 and last[-1][1] == num_points - 1:
                 # Merge last into first (prepend last to first)
                 segments.pop(0)
                 segments.pop(-1)
                 new_seg = last + first
                 segments.append(new_seg)

    return segments


