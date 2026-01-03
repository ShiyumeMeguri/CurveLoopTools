bl_info = {
    "name": "LoopTools Plus",
    "author": "ShiyumeMeguri",
    "version": (0, 2, 1),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Edit Tab / Edit Mode Context Menu (W) / UV Editor Context Menu (W)",
    "description": "LoopTools functionality for Curves and UVs",
    "warning": "",
    "doc_url": "",
    "category": "User",
}

import bpy
from . import operators


class LOOPTOOLSPLUS_MT_menu(bpy.types.Menu):
    bl_label = "LoopTools"
    bl_idname = "LOOPTOOLSPLUS_MT_menu"

    def draw(self, context):
        layout = self.layout
        is_curve = context.active_object and context.active_object.type == 'CURVE'
        is_mesh = context.active_object and context.active_object.type == 'MESH'
        
        if is_curve:
            layout.operator("looptools_plus.curve_circle", text="Circle")
            layout.operator("looptools_plus.curve_flatten", text="Flatten")
            layout.separator()
            
            op_relax = layout.operator("looptools_plus.curve_relax", text="Relax")
            op_relax.relax_position = True
            op_relax.relax_tilt = False
            op_relax.relax_radius = False
            op_relax.opt_lock_length = False 
            op_relax.opt_lock_tilt = False
            op_relax.opt_lock_radius = False
            op_relax.regular = True

            op_tilt = layout.operator("looptools_plus.curve_relax", text="Relax Tilt")
            op_tilt.relax_position = False
            op_tilt.relax_tilt = True
            op_tilt.relax_radius = False
            op_tilt.opt_lock_length = False
            op_tilt.opt_lock_tilt = True
            op_tilt.opt_lock_radius = False

            op_radius = layout.operator("looptools_plus.curve_relax", text="Relax Radius")
            op_radius.relax_position = False
            op_radius.relax_radius = True
            op_radius.relax_tilt = False
            op_radius.opt_lock_length = False
            op_radius.opt_lock_tilt = False
            op_radius.opt_lock_radius = True
            
            layout.operator("looptools_plus.curve_space", text="Space")
            
        elif is_mesh and context.space_data and context.space_data.type == 'IMAGE_EDITOR':
            layout.operator("looptools_plus.uv_circle", text="Circle")
            layout.operator("looptools_plus.uv_flatten", text="Flatten")
            layout.separator()
            layout.operator("looptools_plus.uv_relax", text="Relax")
            layout.operator("looptools_plus.uv_space", text="Space")

def menu_func(self, context):
    self.layout.menu("LOOPTOOLSPLUS_MT_menu")

def register():
    operators.register()
    bpy.utils.register_class(LOOPTOOLSPLUS_MT_menu)
    if hasattr(bpy.types, "VIEW3D_MT_edit_curve_context_menu"):
        bpy.types.VIEW3D_MT_edit_curve_context_menu.prepend(menu_func)
    if hasattr(bpy.types, "IMAGE_MT_uvs_context_menu"):
        bpy.types.IMAGE_MT_uvs_context_menu.prepend(menu_func)

def unregister():
    if hasattr(bpy.types, "IMAGE_MT_uvs_context_menu"):
        bpy.types.IMAGE_MT_uvs_context_menu.remove(menu_func)
    if hasattr(bpy.types, "VIEW3D_MT_edit_curve_context_menu"):
        bpy.types.VIEW3D_MT_edit_curve_context_menu.remove(menu_func)
    bpy.utils.unregister_class(LOOPTOOLSPLUS_MT_menu)
    operators.unregister()
