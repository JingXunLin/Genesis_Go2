import pygame
import numpy as np
from pathlib import Path
class control_gamepad:
    def __init__(self,command_cfg,command_scale=None):
        pygame.init()
        pygame.joystick.init()
        self.use_gamepad = True

        # 获取连接的游戏手柄数量
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            print("no gamepad,open keyboard window")
            self.use_gamepad = False
            screen_width = 300
            screen_height = 480
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            image_path1 = "../picture/keyboard_key.png" #sim2sim
            image_path2 = "picture/keyboard_key.png"
            image_center = (160, 240)
            pygame.display.set_caption("keyboard control")
            image_surface = pygame.Surface((800, 600), pygame.SRCALPHA)
            image_surface.fill((255, 255, 255, 0))  # 透明初始化
            try:
                if Path(image_path1).exists():
                    image_surface = pygame.image.load(image_path1)
                elif Path(image_path2).exists():
                    image_surface = pygame.image.load(image_path2)
                else:
                    print(f"无法加载图片: picture/keyboard_key.png")
            except pygame.error as e:
                print(f"无法加载图片: picture/keyboard_key.png")
                print(e)
                pygame.quit()
                exit()
            image_rect = image_surface.get_rect()
            image_rect.center = image_center
            self.screen.fill((255, 255, 255)) # 背景
            self.screen.blit(image_surface, image_rect)
            pygame.display.flip() # 更新屏幕显示 (一次性) 
        else:
            # 选择第一个手柄
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"link gamepad: {self.joystick.get_name()}")
        self.num_commands = command_cfg.num_commands
        self.target_rad = 0.0
        self.command_cfg = command_cfg
        self.commands = np.zeros(self.num_commands)
        self.command_scale = command_scale
        if self.command_scale is None:
            self.command_scale = [1.0, 1.0, 1.0 ,0.05]
    
    def get_commands(self):
        pygame.event.pump()
        reset_flag = False
        push_robot = False
        if self.use_gamepad:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.JOYBUTTONDOWN:
                    # print(f"按钮 {event.button} 被按下。")
                    match event.button:
                        case 0:
                            reset_flag=True
                # elif event.type == pygame.JOYBUTTONUP:
                #     print(f"按钮 {event.button} 被释放。")
                elif event.type == pygame.JOYAXISMOTION:
                    # print(f"轴 {event.axis},{event.value}")
                    match event.axis:
                        case 1: #ly 前正后负
                            self.commands[0] = -event.value * self.command_scale[0]
                        case 2: #rx 左正右负
                            self.commands[2] = -event.value * self.command_scale[2]
                        # case 3: #lt 增加身高
                        #     self.commands[3] += (event.value+1) * self.command_scale[3]
                        # case 4: #ry 减少身高
                        #     self.commands[3] -= (event.value+1) * self.command_scale[3]
        else:
            quiet_walking = 1.0
            keys = pygame.key.get_pressed()  # 取得目前所有按鍵的狀態
            if keys[pygame.K_q]:
                self.commands[2] += self.command_scale[2] * quiet_walking
            if keys[pygame.K_e]:
                self.commands[2] -= self.command_scale[2] * quiet_walking
            
            for event in pygame.event.get():  # 获取事件队列中的所有事件
                if event.type == pygame.QUIT:  # 用户点击窗口关闭按钮
                    running = False
                elif event.type == pygame.KEYDOWN:  # 键盘按键按下事件
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] = self.command_scale[0] * quiet_walking
                        case pygame.K_s:
                            self.commands[0] = -self.command_scale[0] * quiet_walking
                        case pygame.K_a:
                            self.commands[1] = self.command_scale[1] * quiet_walking
                        case pygame.K_d:
                            self.commands[1] = -self.command_scale[1] * quiet_walking
                        # case pygame.K_q:
                        #     self.commands[2] += self.command_scale[2] * quiet_walking
                        # case pygame.K_e:
                        #     self.commands[2] += -self.command_scale[2] * quiet_walking
                        # case pygame.K_SPACE:
                        #     self.commands[3] = +self.command_scale[3] * 100.0
                        # case pygame.K_c:
                        #     self.commands[3] = -self.command_scale[3] * 100.0
                        # case pygame.K_LCTRL:
                        #     self.commands[3] = -self.command_scale[3] * 100.0
                        case pygame.K_SPACE:
                            push_robot = True
                        case pygame.K_r:
                            reset_flag=True
                        case pygame.K_LSHIFT:
                            quiet_walking=0.2
                elif event.type == pygame.KEYUP:  # 键盘按键释放事件
                    match event.key:
                        case pygame.K_w:
                            self.commands[0] = 0
                        case pygame.K_s:
                            self.commands[0] = 0
                        case pygame.K_a:
                            self.commands[1] = 0
                        case pygame.K_d:
                            self.commands[1] = 0
                        # case pygame.K_q:
                        #     self.commands[2] = 0
                        # case pygame.K_e:
                        #     self.commands[2] = 0
        self.commands_clip()
        return self.commands,reset_flag, push_robot
    
    def commands_clip(self):
        # lin_vel_x
        if self.commands[0] < self.command_cfg.ranges.lin_vel_x[0] * self.command_scale[0]:
            self.commands[0] = self.command_cfg.ranges.lin_vel_x[0] * self.command_scale[0]
        elif self.commands[0] > self.command_cfg.ranges.lin_vel_x[1] * self.command_scale[0]:
            self.commands[0] = self.command_cfg.ranges.lin_vel_x[1] * self.command_scale[0]

        #lin_vel_y
        if self.commands[1] < self.command_cfg.ranges.lin_vel_y[0] * self.command_scale[1]:
            self.commands[1] = self.command_cfg.ranges.lin_vel_y[0] * self.command_scale[1]
        elif self.commands[1] > self.command_cfg.ranges.lin_vel_y[1] * self.command_scale[1]:
            self.commands[1] = self.command_cfg.ranges.lin_vel_y[1] * self.command_scale[1]

        #target_rad
        while self.commands[2] >= 6.28:
            self.commands[2] -= 6.28
        #ang_vel
        # if self.commands[2] < self.command_cfg.ranges.ang_vel_yaw[0] * self.command_scale[2]:
        #     self.commands[2] = self.command_cfg.ranges.ang_vel_yaw[0] * self.command_scale[2]
        # elif self.commands[2] > self.command_cfg.ranges.ang_vel_yaw[1] * self.command_scale[2]:
        #     self.commands[2] = self.command_cfg.ranges.ang_vel_yaw[1] * self.command_scale[2]

        #base_heigh
        # if self.commands[3] < self.command_cfg["height_target_range"][0]:
        #     self.commands[3] = self.command_cfg["height_target_range"][0]
        # elif self.commands[3] > self.command_cfg["height_target_range"][1]:
        #     self.commands[3] = self.command_cfg["height_target_range"][1]