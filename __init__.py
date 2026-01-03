bl_info = {
    "name": "Curve LoopTools",
    "author": "ShiyumeMeguri",
    "version": (0, 1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Edit Tab / Edit Mode Context Menu (W)",
    "description": "LoopTools functionality for Curves",
    "warning": "",
    "doc_url": "",
    "category": "Curve",
}

import bpy
from . import operators


class CURVELOOPTOOLS_MT_menu(bpy.types.Menu):
    bl_label = "LoopTools"
    bl_idname = "CURVELOOPTOOLS_MT_menu"

    def draw(self, context):
        layout = self.layout
        # layout.operator("curve_looptools.bridge", text="Bridge")
        layout.operator("curve_looptools.circle", text="Circle")
        # layout.operator("curve_looptools.curve", text="Curve")
        layout.operator("curve_looptools.flatten", text="Flatten")
        # layout.operator("curve_looptools.gstretch", text="Gstretch")
        # layout.operator("curve_looptools.loft", text="Loft")
        layout.separator()
        
        op_relax = layout.operator("curve_looptools.relax", text="Relax")
        op_relax.relax_position = True
        op_relax.relax_tilt = False
        op_relax.relax_radius = False
        # Do not force lock_length, let user decide? Or force default?
        # User said: "default options override... conflict ones"
        # If I don't set it, it uses default from operator definition (False).
        # User wants "Relax" to be standard relax.
        op_relax.opt_lock_length = False 
        op_relax.opt_lock_tilt = False
        op_relax.opt_lock_radius = False
        op_relax.regular = False

        
        op_tilt = layout.operator("curve_looptools.relax", text="Relax Tilt")
        op_tilt.relax_position = False
        op_tilt.relax_tilt = True
        op_tilt.relax_radius = False
        op_tilt.opt_lock_length = False
        op_tilt.opt_lock_tilt = True
        op_tilt.opt_lock_radius = False

        op_radius = layout.operator("curve_looptools.relax", text="Relax Radius")
        op_radius.relax_position = False
        op_radius.relax_radius = True
        op_radius.relax_tilt = False
        op_radius.opt_lock_length = False
        op_radius.opt_lock_tilt = False
        op_radius.opt_lock_radius = True
        
        layout.operator("curve_looptools.space", text="Space")

def menu_func(self, context):
    self.layout.menu("CURVELOOPTOOLS_MT_menu")

def register():
    operators.register()
    bpy.utils.register_class(CURVELOOPTOOLS_MT_menu)
    bpy.types.VIEW3D_MT_edit_curve_context_menu.prepend(menu_func)

def unregister():
    bpy.types.VIEW3D_MT_edit_curve_context_menu.remove(menu_func)
    bpy.utils.unregister_class(CURVELOOPTOOLS_MT_menu)
    operators.unregister()
