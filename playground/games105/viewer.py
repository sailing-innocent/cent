from direct.showbase.ShowBase import ShowBase
import numpy as np
from panda3d.core import ClockObject
import panda3d.core as pc
import math
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *

class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.scene = self.loader.loadModel("models/box")
        self.scene.reparentTo(self.render)
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

class CameraCtrl(DirectObject):
    def __init__(self, base, camera):
        super(CameraCtrl).__init__()
        # self.accept('mouse1',self.onMouse1Down)
        # self.accept('mouse1-up',self.onMouse1Up)
        # self.accept('mouse2',self.onMouse2Down)
        # self.accept('mouse2-up',self.onMouse2Up)
        # self.accept('mouse3',self.onMouse3Down)
        # self.accept('mouse3-up',self.onMouse3Up)
        # self.accept('wheel_down',self.onMouseWheelDown)
        # self.accept('wheel_up',self.onMouseWheelUp)

        # self.accept('control-mouse1',self.onMouse1Down)
        # self.accept('control-mouse1-up',self.onMouse1Up)
        # self.accept('control-mouse2',self.onMouse2Down)
        # self.accept('control-mouse2-up',self.onMouse2Up)
        # self.accept('control-mouse3',self.onMouse3Down)
        # self.accept('control-mouse3-up',self.onMouse3Up)
        # self.accept('control-wheel_down',self.onMouseWheelDown)
        # self.accept('control-wheel_up',self.onMouseWheelUp)

        self.position = pc.LVector3(4,4,4)
        self.center = pc.LVector3(0,1,0)
        self.up = pc.LVector3(0,1,0)
        self.base = base
        # base.taskMgr.add(self.onUpdate, 'updateCamera')
        self.camera = camera
        self._locked_info = None
        self._locked_mouse_pos = None
        self._mouse_id = -1
        self.look()
    
    def look(self):
        self.camera.setPos(self.position)
        self.camera.lookAt(self.center, self.up)


class SimpleViewer(ShowBase):
    def __init__(self, fStartDirect=True, windowType=None):
        super().__init__(fStartDirect, windowType)
        self.disableMouse()

        self.camera.lookAt(0, 0.9, 0)
        self.setupCameraLight()
        self.camera.setHpr(0, 0, 0)

        # self.setFrameRateMeter(True)
        # globalClock.setMode(ClockObject.MLimited)
        # globalClock.setFrameRate(60)

        self.load_ground()

        xSize = self.pipe.getDisplayWidth()
        ySize = self.pipe.getDisplayHeight()
        props = pc.WindowProperties()
        props.setSize(min(xSize-200,800), min(ySize-200,600))
        self.win.requestProperties(props)

        # color for links
        color = [131/255, 175/255, 155/255, 1]
        # self.tex = self.create_texture(color, 'link_tex')

        self.load_character()

        self.load_character()
        self.update_func = None
        self.add_task(self.update, 'update')
        self.update_flag = True 
        self.accept('space', self.receive_space)

        pass

    def load_ground(self):
        self.ground = self.loader.loadModel("material/GroundScene.egg")
        self.ground.reparentTo(self.render)
        self.ground.setScale(100, 1, 100)
        self.ground.setTexScale(pc.TextureStage.getDefault(), 50, 50)
        self.ground.setPos(0, -1, 0)

    def receive_space():
        pass
    
    def create_texture():
        pass

    def setupCameraLight(self):
        self.cameractrl = CameraCtrl(self, self.cam)
        self.cameraRefNode = self.camera
        self.cameraRefNode.setPos(0,0,0)
        self.cameraRefNode.setHpr(0,0,0)
        self.cameraRefNode.reparentTo(self.render)

        # ambient light
        # directional light 01
        # directional light 02

    def create_joint(self, link_id, position, end_effector=False):
        box = self.loader.loadModel("material/GroundScene.egg")
        node = self.render.attachNewNode(f"joint{link_id}")
        box.reparentTo(node)
    
    def update(self, task):
        if (self.update_func and self.update_flag):
            self.update_func(self)
        return task.cont
    
    def load_character(self):
        info = np.load('character_model.npy', allow_pickle=True).item()
        joint_pos = info['joint_pos']
        print(joint_pos)