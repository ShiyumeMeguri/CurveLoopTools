bl_info = {
    "name": "Curve LoopTools",
    "author": "Antigravity",
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

class CURVELOOPTOOLS_PT_main(bpy.types.Panel):
    bl_label = "Curve LoopTools"
    bl_idname = "CURVELOOPTOOLS_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Edit"
    bl_context = "curve_edit"

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Tools")
        
        # row = col.row(align=True)
        # row.operator("curve_looptools.bridge", text="Bridge") # Not impl
        # row.operator("curve_looptools.circle", text="Circle") # Impl
        
        row = col.row(align=True)
        # row.operator("curve_looptools.curve", text="Curve") # Not impl
        row.operator("curve_looptools.flatten", text="Flatten")
        row.operator("curve_looptools.circle", text="Circle")
        
        # row = col.row(align=True)
        # row.operator("curve_looptools.gstretch", text="Gstretch") # Not impl
        # row.operator("curve_looptools.loft", text="Loft") # Not impl
        col.separator()
        row = col.row(align=True)
        row.operator("curve_looptools.relax", text="Relax").relax_position = True
        row.operator("curve_looptools.space", text="Space")
        
        row = col.row(align=True)
        op_tilt = row.operator("curve_looptools.relax", text="Relax Tilt")
        op_tilt.relax_position = False
        op_tilt.relax_tilt = True
        
        op_radius = row.operator("curve_looptools.relax", text="Relax Radius")
        op_radius.relax_position = False
        op_radius.relax_radius = True

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
        op_relax.lock_length = True
        
        op_tilt = layout.operator("curve_looptools.relax", text="Relax Tilt")
        op_tilt.relax_position = False
        op_tilt.relax_tilt = True
        op_tilt.relax_radius = False
        op_tilt.lock_tilt = True

        op_radius = layout.operator("curve_looptools.relax", text="Relax Radius")
        op_radius.relax_position = False
        op_radius.relax_radius = True
        op_radius.relax_tilt = False
        op_radius.lock_radius = True
        
        layout.operator("curve_looptools.space", text="Space")

def menu_func(self, context):
    self.layout.menu("CURVELOOPTOOLS_MT_menu")

def register():
    operators.register()
    bpy.utils.register_class(CURVELOOPTOOLS_PT_main)
    bpy.utils.register_class(CURVELOOPTOOLS_MT_menu)
    bpy.types.VIEW3D_MT_edit_curve_context_menu.prepend(menu_func)

def unregister():
    bpy.types.VIEW3D_MT_edit_curve_context_menu.remove(menu_func)
    bpy.utils.unregister_class(CURVELOOPTOOLS_MT_menu)
    bpy.utils.unregister_class(CURVELOOPTOOLS_PT_main)
    operators.unregister()
