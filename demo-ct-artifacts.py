# Copyright 2024 tsadakane
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

'''
Generate animations of the image generation process 
of CT reconstruction that aid in the intuitive 
understanding of artifacts in CT reconstruction.
'''
import os
import time
import threading
import numpy as np
import tkinter as tk
import PIL.Image
import PIL.ImageOps
import PIL.ImageTk
import skimage.data
import skimage.transform

VIEW_SIZE = 200
CANVAS_SIZE = 450
FPS = 30

class App(tk.Frame):
    _img_sinogram: PIL.ImageTk.PhotoImage
    _img_recon_wip: PIL.ImageTk.PhotoImage
    _img_recon: PIL.ImageTk.PhotoImage
    _ip_flag = np.array([1,0,0,0])
    _index = 0
    _fast_step_size = 20
    def __init__(self, master, view_size, canvas_size, fps):
        self._FPS = fps
        self._view_size = view_size
        self._canvas_size = canvas_size

        self._is_playing = True
        self._index = 0
        self._list_thread_busy = None
        self._terminate_thread = False
        self._sinogram_updated = True

        self._canvas_image_id_sinogram = None
        self._canvas_image_id_reconned = None
        self._sinogram_line_id = None

        super().__init__(master)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.master.title("Artifacts on CT images")
        self.master.geometry("{}x{}".format((self._canvas_size + 10)*3, self._canvas_size+125))

        fonts=("Times New Roman", 12)

        image = skimage.data.shepp_logan_phantom()
        self._sl_image = skimage.transform.rescale(image, scale=1, mode='reflect', channel_axis=None)
        self._nangles = max(image.shape)
        self._list_img_recon = [None]*self._nangles
        self._list_img_bp = [None]*self._nangles

        self._angle_defect = None
        self._pixel_defect = None
        self._half_scan = False

        self._xray_intensity = "Infinity"

        self.reset_images()

        frame_imgs = tk.Frame(self.master)
        frame_imgs.grid(row = 0, column=0, columnspan=3)
        self._canvas_sinogram = tk.Canvas(frame_imgs, width = self._canvas_size, height=self._canvas_size)
        self._canvas_progress = tk.Canvas(frame_imgs, width = self._canvas_size, height=self._canvas_size)
        self._canvas_reconned = tk.Canvas(frame_imgs, width = self._canvas_size, height=self._canvas_size)
        self._canvas_sinogram.grid(row=0, column=0)
        self._canvas_progress.grid(row=0, column=1)
        self._canvas_reconned.grid(row=0, column=2)

        frame_play = tk.Frame(self.master)
        frame_play.grid(row = 1, column=0)
        frame_proj = tk.Frame(self.master)
        frame_proj.grid(row = 1, column=1)
        frame_xray = tk.Frame(self.master)
        frame_xray.grid(row = 1, column=2)

        self._btn_defect_pix = tk.Button(frame_proj, font=fonts, text="Defect Pixel", command = self.on_pixel_defect)
        self._btn_defect_prj = tk.Button(frame_proj, font=fonts, text="Defect Proj.", command = self.on_angle_defect)
        self._btn_proj_range = tk.Button(frame_proj, font=fonts, text="180/360", command = self.on_half_scan)
        self._lbl_proj_desc = tk.Label(frame_proj, text=self._get_label_text())
        self._btn_defect_pix.grid(row=0, column=0)
        self._btn_defect_prj.grid(row=1, column=0)
        self._btn_proj_range.grid(row=2, column=0)
        self._lbl_proj_desc.grid(row=3, column=0)
        self._btn_xray_int_Infty = tk.Button(frame_xray, font=fonts, text="Infty", command = self.on_xray_intensity_inf)
        self._btn_xray_int_10_2 = tk.Button(frame_xray, font=fonts, text="100", command = self.on_xray_intensity_100)
        self._btn_xray_int_10_3 = tk.Button(frame_xray, font=fonts, text="1000", command = self.on_xray_intensity_1000)
        self._btn_xray_int_10_4 = tk.Button(frame_xray, font=fonts, text="10000", command = self.on_xray_intensity_10000)
        self._btn_xray_int_10_5 = tk.Button(frame_xray, font=fonts, text="100000", command = self.on_xray_intensity_100000)
        self._lbl_xray = tk.Label(frame_xray, text=self._get_label_text_xray())
        self._btn_xray_int_10_2.grid(row=0, column=0)
        self._btn_xray_int_10_3.grid(row=0, column=1)
        self._btn_xray_int_10_4.grid(row=0, column=2)
        self._btn_xray_int_10_5.grid(row=0, column=3)
        self._btn_xray_int_Infty.grid(row=0, column=4)
        self._lbl_xray.grid(row=1, column=0, columnspan=5)

        self._btn_step_b = tk.Button(frame_play, font=fonts, text="<", command = self.on_play_step_b)
        self._btn_playpause = tk.Button(frame_play, font=fonts, text="Play/Pause", command = self.on_play_pause)
        self._btn_step_f = tk.Button(frame_play, font=fonts, text=">", command = self.on_play_step_f)
        self._btn_fast_b = tk.Button(frame_play, font=fonts, text="<<", command = self.on_play_step_fastb)
        self._btn_fast_f = tk.Button(frame_play, font=fonts, text=">>", command = self.on_play_step_fastf)
        self._btn_playpause.grid(row=0, column=2)
        self._btn_step_b.grid(row=0, column=1)
        self._btn_step_f.grid(row=0, column=3)
        self._btn_fast_b.grid(row=0, column=0)
        self._btn_fast_f.grid(row=0, column=4)
        self._lbl_current_index = tk.Label(frame_play, text=self._get_current_index_text())
        self._lbl_current_index.grid(row=1, column=0, columnspan=5)
        self._lbl_current_angle = tk.Label(frame_play, text=self._get_current_angle_text())
        self._lbl_current_angle.grid(row=2, column=0, columnspan=5)

        self.display_image()

    def _ndarray_to_photoimage(self, ary: np.ndarray, shape):
        tmp = PIL.Image.fromarray(ary)
        tmp = PIL.ImageOps.pad(tmp, shape)
        tmp = PIL.ImageTk.PhotoImage(image = tmp)
        return tmp

    def _get_label_text_xray(self):
        return f"X-ray Intensity: {self._xray_intensity}"

    def _get_label_text(self):
        str_pix = "Pixel Defect, " if self._pixel_defect is not None else ""
        str_angle = "Proj Defect, " if self._angle_defect is not None else ""
        str_half = "Half Scan" if self._half_scan else "Full Scan"
        str_xray_intensity = f"X-ray Intensity: {self._xray_intensity}"
        return str_pix + str_angle + str_half + " " + str_xray_intensity
    
    def _get_current_index_text(self):
        return f"Projection ID: {self._index}/{self._nangles}"
    
    def _get_current_angle_text(self):
        deg_angle = self._index/self._nangles * 180. * (1 if self._half_scan else 2)
        return f"Angle: {deg_angle:.1f} deg."
    
    def update_angles(self):
        """
            update angles
        """
        if self._half_scan == False:
            angle_max = 360.
        else:
            angle_max = 180.
        self._angles = np.linspace(0., angle_max, self._nangles, endpoint=False)

    def update_sinogram(self):
        """ 
            update sinogram
        """
        self._sinogram = skimage.transform.radon(self._sl_image, theta=self._angles)
        print(f"sinogram size = {self._sinogram.shape}")
        if self._xray_intensity != "Infinity":
            I0 = float(self._xray_intensity)
            self._sinogram = self._add_poisson_noise(self._sinogram, I0)
        if self._angle_defect != None:
            for idx in range(len(self._angle_defect)):
                self._sinogram[:, self._angle_defect[idx]] = 0
            # self._sinogram[:, self._angle_defect] = 0
        if self._pixel_defect != None:
            for idx in range(len(self._pixel_defect)):
                self._sinogram[self._pixel_defect[idx], :] = 0
        
        self._reconned = skimage.transform.iradon(self._sinogram, theta=self._angles, filter_name='ramp')*255

    def _add_poisson_noise(self, sino, I0 = 100):
        '''
            Add Poisson noise to sinogram
        '''
        rng = np.random.default_rng()
        # Resore the original intensity
        proj = np.exp(-sino/255) * I0
        # Add Poisson noise
        proj = rng.poisson(proj)
        # Convert back to sinogram
        sino = -np.log(proj/I0) * 255
        # clip negative values
        sino[sino < 0] = 0
        return sino
        
    def display_image(self):

        if self._sinogram_updated:
            sino = self._sinogram
            self._img_sinogram = self._ndarray_to_photoimage(sino, shape=self._sinogram.shape)
            self._img_recon = self._ndarray_to_photoimage(self._reconned, shape=self._reconned.shape)

        if self._canvas_image_id_sinogram == None:
            self._canvas_image_id_sinogram = self._canvas_sinogram.create_image(0, 0, image = self._img_sinogram, anchor='nw')
            self._canvas_image_id_reconned = self._canvas_reconned.create_image(0, 0, image = self._img_recon, anchor='nw')

        self._canvas_sinogram.itemconfigure(self._canvas_image_id_sinogram, image=self._img_sinogram)
        self._canvas_reconned.itemconfigure(self._canvas_image_id_reconned, image=self._img_recon)
        self._sinogram_updated = False
        self._lbl_current_index["text"] = self._get_current_index_text()
        self._lbl_current_angle["text"] = self._get_current_angle_text()
        
        if self._sinogram_line_id is not None:
            self._canvas_sinogram.delete(self._sinogram_line_id)
        self._sinogram_line_id = self._canvas_sinogram.create_line(self._index, 0, self._index, 399, fill ="red")

        if self._list_img_recon[self._index] is not None:
            img_recon = self._list_img_recon[self._index]
            self._img_recon_wip = self._ndarray_to_photoimage(img_recon, shape=img_recon.shape)
            self._canvas_progress.create_image(0, 0, image = self._img_recon_wip, anchor='nw')
        else:
            self._index -=1
        
        if self._is_playing:
            self._step_proj(1)

        self.after(int(1000/self._FPS), self.display_image)

    def _step_proj(self, inc):
        self._index += inc
        if self._index >= self._nangles:
            self._index -= self._nangles
        if self._index < 0:
            self._index += self._nangles

    def reset_images(self):
        self._terminate_thread = True
        self.update_angles()
        self.update_sinogram()
        self._sinogram_updated = True
        self._list_img_recon = [None]*self._nangles
        self._index = 0
        self._is_playing = True
        method = 1
        if method == 0:
            thread_count = min(4, max(os.cpu_count()-2, 1))
            self._list_thread_busy = [False]*thread_count
            for idx in range(thread_count):
                thread = threading.Thread(target=self.reconstruct_images_0, args=(idx, thread_count))
                thread.start()
        else:
            thread_count = min(4, max(os.cpu_count()-2, 1))
            self._list_thread_busy = [False]*thread_count
            step = self._nangles // thread_count
            for idx in range(thread_count):
                idx_begin = idx*step
                idx_end = (idx+1)*step if idx != thread_count-1 else self._nangles 
                thread = threading.Thread(target=self.reconstruct_images_1, args=(idx, idx_begin, idx_end))
                thread.start()          

    def reconstruct_images_0(self, idx_ini, idx_step):
        self._list_thread_busy[idx_ini] = True
        self._terminate_thread = False
        for idx in range(idx_ini, self._nangles, idx_step):
            if self._terminate_thread:
                break
            self._list_img_recon[idx] = skimage.transform.iradon(self._sinogram[:, 0:(idx+1)], theta=self._angles[0:(idx+1)], filter_name='ramp')*255
            # print(f"thread {idx_ini}:{idx}")
        self._list_thread_busy[idx_ini] = False
            
    def reconstruct_images_1(self, idx_th, idx_begin, idx_end):
        self._list_thread_busy[idx_th] = True
        self._terminate_thread = False
        if idx_begin != 0:
            img_base = skimage.transform.iradon(self._sinogram[:, 0:idx_begin]
                          , theta=self._angles[0:idx_begin]
                          , filter_name='ramp') * idx_begin
            self._list_img_recon[idx_begin] = img_base / idx_begin *255
        else:
            img_base = np.zeros(shape=(self._sinogram.shape[0], self._sinogram.shape[0]), dtype = self._sinogram.dtype)
            self._list_img_recon[idx_begin] = img_base

        for idx in range(idx_begin, idx_end):
            if self._terminate_thread:
                break
            img_sinble_bp = skimage.transform.iradon(self._sinogram[:, idx:(idx+1)], theta=self._angles[idx:(idx+1)], filter_name='ramp')
            img_base += img_sinble_bp
            self._list_img_recon[idx] = img_base / idx * 255
            # print(f"thread {idx_ini}:{idx}")
        self._list_thread_busy[idx_th] = False

    def on_xray_intensity_inf(self):
        self._on_xray_intensity_common("Infinity") 

    def on_xray_intensity_100(self):
        self._on_xray_intensity_common("100")

    def on_xray_intensity_1000(self):
        self._on_xray_intensity_common("1000")

    def on_xray_intensity_10000(self):
        self._on_xray_intensity_common("10000")

    def on_xray_intensity_100000(self):
        self._on_xray_intensity_common("100000")

    def _on_xray_intensity_common(self, intensity):
        self._xray_intensity = intensity
        self._lbl_xray.config(text=self._get_label_text_xray())
        self.reset_images()

    def on_pixel_defect(self):
        if self._pixel_defect:
            self._pixel_defect = None
        else:
            self._pixel_defect = [slice( 100, 101), slice( 220, 221)]
        self._lbl_proj_desc.config(text=self._get_label_text())
        self.reset_images()

    def on_angle_defect(self):
        if self._angle_defect:
            self._angle_defect = None
        else:
            self._angle_defect = [slice( 45, 46), ]
        self._lbl_proj_desc.config(text=self._get_label_text())
        self.reset_images()

    def on_half_scan(self):
        if self._half_scan:
            self._half_scan = False
        else:
            self._half_scan = True
        self._lbl_proj_desc.config(text=self._get_label_text())
        self.reset_images()

    def on_play_pause(self):
        if self._is_playing:
            self._is_playing = False
        else:
            self._is_playing = True

    def on_play_step_f(self):
        if not self._is_playing:
            self._step_proj(1)

    def on_play_step_b(self):
        if not self._is_playing:
            self._step_proj(-1)

    def on_play_step_fastf(self):
        self._step_proj(self._fast_step_size)

    def on_play_step_fastb(self):
        self._step_proj(-self._fast_step_size)

        
    def on_closing(self):
        self._terminate_thread = True
        while True:
            if all(self._list_thread_busy) == False:
                break
            else:
                time.sleep(1)
        self.master.destroy()

app = App(tk.Tk(), VIEW_SIZE, CANVAS_SIZE, FPS)
app.mainloop()
