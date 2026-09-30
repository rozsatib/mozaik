r"""
This module contains API of visual stimuli.
"""
import numpy
import mozaik
from quantities import degrees
from mozaik.stimuli import BaseStimulus
from mozaik.space import TRANSPARENT, xy2ij, VisualRegion
from mozaik.tools.mozaik_parametrized import SNumber
from mozaik.tools.units import lux
from scipy.ndimage import interpolation
from collections import OrderedDict
from mozaik.tools.mozaik_parametrized import SNumber, SString, SParameterSet
from mozaik.tools.distribution_parametrization import MozaikExtendedParameterSet
from quantities import Hz, rad, degrees, ms, dimensionless

logger = mozaik.getMozaikLogger()


class VisualStimulus(BaseStimulus):
    r"""
    Abstract base class for visual stimuli.
    
    This class defines all parameters common to all visual stimuli.
    
    This class implements all functions specified by the :class:`mozaik.stimuli.stimulus.BaseStimulus` interface.
    The only function that remains to be implemented by the user whenever creating a new stimulus by subclassing
    this class is the :func:`mozaik.stimuli.stimulus.BaseStimulus.frames` function.
    
    This class also implements functions common to all visual stimuli that are required for it to be compatible
    with the :class:`mozaik.space.VisualSpace` and  class.

    Frames can be yielded either as 2D arrays or as deferred frames, which are
    rendered only when needed. A subclass that yields deferred frames implements
    :func:`_render_frame` and :func:`_frame_shape`. :func:`display` then renders
    only the pixel window it was asked for, as long as :func:`_renders_windows`
    allows it for the frame and the windows rendered in a frame stay smaller than
    the whole frame; otherwise the whole frame is rendered once and cropped. Both
    paths must produce the same pixels.
    """
    background_luminance = SNumber(lux, doc="Background luminance. Maximum luminance of object allowed is 2*background_luminance")
    density = SNumber(1/(degrees), doc="The density of stimulus - units per degree")
    location_x = SNumber(degrees, doc="x location of the center of  visual region.")
    location_y = SNumber(degrees, doc="y location of the center of  visual region.")
    size_x = SNumber(degrees, doc="The size of the region in degrees (asimuth).")
    size_y = SNumber(degrees, doc="The size of the region in degrees (elevation).")

    # Set to False to always render whole frames, e.g. to compare both paths.
    window_rendering = True

    def __init__(self, **params):
        BaseStimulus.__init__(self, **params)
        self._zoom_cache = OrderedDict()
        self.region_cache = OrderedDict()
        self._img = None
        self._deferred_frame = None
        self._window_pixels = 0
        self._render_whole_frames = not self.window_rendering
        self.is_visible = True
        self.transparent = True # And efficiency flag. It should be set to false by the stimulus if there are no transparent points in it. 
                                # This will avoid all the code related to transparency which is very expensive.
        self.region = VisualRegion(self.location_x, self.location_y,
                                   self.size_x, self.size_y)
        self.first_resolution_mismatch_display=True
        

    def _calculate_zoom(self, actual_pixel_size, desired_pixel_size):
        r"""
        Sometimes the interpolation procedure returns a new array that is too
        small due to rounding error. This is a crude attempt to work around that.
        """
        zoom = actual_pixel_size/desired_pixel_size
        for i in self._current_frame_shape():
            if int(zoom*i) != round(zoom*i):
                zoom *= (1 + 1e-15)
        return zoom

    def display(self, region, pixel_size):
        assert isinstance(region, VisualRegion), "region must be a VisualRegion-descended object. Actually a %s" % type(region)
        size_in_pixels = numpy.ceil(
                            xy2ij((region.size_x, region.size_y))
                                /float(pixel_size)).astype(int)
        if self.transparent:
            view = TRANSPARENT * numpy.ones(size_in_pixels)
        else:
            view = self.background_luminance * numpy.ones(size_in_pixels)
            
        if region.overlaps(self.region):
            if region not in self.region_cache:
                intersection = region.intersection(self.region)
                assert intersection == self.region.intersection(region)  # just a consistency check. Could be removed if necessary for performance.
                img_relative_left = (intersection.left - self.region.left) / self.region.width
                img_relative_width = intersection.width/self.region.width
                img_relative_top = (intersection.top - self.region.bottom) / self.region.height
                #img_relative_bottom = (intersection.bottom - self.region.bottom) / self.region.height
                img_relative_height = intersection.height / self.region.height
                view_relative_left = (intersection.left - region.left) / region.width
                view_relative_width = intersection.width / region.width
                view_relative_top = (intersection.top - region.bottom) / region.height
                #view_relative_bottom = (intersection.bottom - region.bottom) / region.height
                view_relative_height = intersection.height / region.height

                img_shape = self._current_frame_shape()
                img_pixel_size = xy2ij((self.region.size_x, self.region.size_y)) / img_shape  # is self.size a tuple or an array?
                assert img_pixel_size[0] == img_pixel_size[1]

                # necessary instead of == comparison due to the floating math rounding errors
                if abs(pixel_size-img_pixel_size[0])<0.01:
                    zoom = None
                else:
                    if self.first_resolution_mismatch_display:
                        logger.warning("Image pixel size does not match desired size (%g vs. %g) degrees. This is extremely inefficient!!!!!!!!!!!" % (pixel_size,img_pixel_size[0]))
                        logger.warning("Image pixel size %g,%g" % numpy.shape(self.img))
                        self.first_resolution_mismatch_display = False
                    # note that if the image is much larger than the view region, we might save some
                    # time by not rescaling the whole image, only the part within the view region.
                    zoom = self._calculate_zoom(img_pixel_size[0], pixel_size)  # img_pixel_size[0]/pixel_size
                    #logger.debug("Image pixel size (%g deg) does not match desired size (%g deg). Zooming image by a factor %g" % (img_pixel_size[0], pixel_size, zoom))
                    if zoom not in self._zoom_cache:
                        self._zoom_cache[zoom] = interpolation.zoom(self.img, zoom)
                    img_shape = self._zoom_cache[zoom].shape

                j_start = numpy.round(img_relative_left * img_shape[1]).astype(int)
                delta_j = numpy.round(img_relative_width * img_shape[1]).astype(int)
                i_start = img_shape[0] - numpy.round(img_relative_top * img_shape[0]).astype(int)
                delta_i = numpy.round(img_relative_height * img_shape[0]).astype(int)

                l_start = numpy.round(view_relative_left * size_in_pixels[1]).astype(int)
                delta_l = numpy.round(view_relative_width * size_in_pixels[1]).astype(int)
                k_start = size_in_pixels[0] - numpy.round(view_relative_top * size_in_pixels[0]).astype(int)
                delta_k = numpy.round(view_relative_height * size_in_pixels[0]).astype(int)
                
                # unfortunatelly the above code can give inconsistent results even if the inputs are correct due to rounding errors
                # therefore:
                
                if abs(delta_j-delta_l) == 1:
                   delta_j = min(delta_j,delta_l)
                   delta_l = min(delta_j,delta_l)

                if abs(delta_i-delta_k) == 1:
                   delta_i = min(delta_i,delta_k)
                   delta_k = min(delta_i,delta_k)
                
                assert delta_j == delta_l, "delta_j = %g, delta_l = %g" % (delta_j, delta_l)
                assert delta_i == delta_k, "delta_i = %g, delta_k = %g" % (delta_i, delta_k)

                i_stop = i_start + delta_i
                j_stop = j_start + delta_j
                k_stop = k_start + delta_k
                l_stop = l_start + delta_l
                ##logger.debug("i_start = %d, i_stop = %d, j_start = %d, j_stop = %d" % (i_start, i_stop, j_start, j_stop))
                ##logger.debug("k_start = %d, k_stop = %d, l_start = %d, l_stop = %d" % (k_start, k_stop, l_start, l_stop))

                # The zoom is cached with the slices, as they index the zoomed image.
                self.region_cache[region] = ((k_start,k_stop,l_start,l_stop),(i_start,i_stop, j_start,j_stop),zoom)

            ((k_start,k_stop,l_start,l_stop),(i_start,i_stop,j_start,j_stop),zoom) = self.region_cache[region]
            pixels = self._frame_pixels(i_start, i_stop, j_start, j_stop, zoom)
            try:
                view[k_start:k_stop, l_start:l_stop] = pixels
            except ValueError:
                logger.error("i_start = %d, i_stop = %d, j_start = %d, j_stop = %d" % (i_start, i_stop, j_start, j_stop))
                logger.error("k_start = %d, k_stop = %d, l_start = %d, l_stop = %d" % (k_start, k_stop, l_start, l_stop))
                logger.error("frame shape = %s, view.shape = %s" % (self._current_frame_shape(), view.shape))
                logger.error("frame[i_start:i_stop, j_start:j_stop].shape = %s" % str(pixels.shape))
                logger.error("view[k_start:k_stop, l_start:l_stop].shape = %s" % str(view[k_start:k_stop, l_start:l_stop].shape))
                raise
        return view

    def _renders_windows(self, frame):
        r"""
        Whether the deferred `frame` can be rendered one pixel window at a time.
        Subclasses that yield deferred frames override this.
        """
        return False

    def _frame_shape(self, frame):
        r"""
        The pixel shape of the whole deferred `frame`, obtained without rendering it.
        """
        raise NotImplementedError

    def _render_frame(self, frame, window=None):
        r"""
        Render the deferred `frame` as a 2D array: the whole frame, or only the pixel
        window (i_start, i_stop, j_start, j_stop) of the whole frame.
        """
        raise NotImplementedError

    def _current_frame_shape(self):
        if self._img is None and self._deferred_frame is not None:
            return tuple(self._frame_shape(self._deferred_frame))
        return self.img.shape

    def _frame_pixels(self, i_start, i_stop, j_start, j_stop, zoom):
        r"""
        Pixels [i_start:i_stop, j_start:j_stop] of the current frame, zoomed by `zoom` unless it is None.
        """
        if zoom is not None:
            if zoom not in self._zoom_cache:
                self._zoom_cache[zoom] = interpolation.zoom(self.img, zoom)
            return self._zoom_cache[zoom][i_start:i_stop, j_start:j_stop]
        frame = self._deferred_frame
        if (self._img is None and frame is not None and not self._render_whole_frames
                and self._renders_windows(frame)):
            window = self._check_luminance(self._render_frame(frame, (i_start, i_stop, j_start, j_stop)))
            self._window_pixels += window.size
            if self._window_pixels > numpy.prod(self._frame_shape(frame)):
                # The windows now cost more than the whole frame (e.g. the viewed regions cover
                # the field), so render whole frames for the rest of this stimulus.
                self._render_whole_frames = True
            return window
        return self.img[i_start:i_stop, j_start:j_stop]

    def _check_luminance(self, frame):
        assert frame.min() >= 0 or frame.min() == TRANSPARENT, "frame minimum is less than zero: %g" % frame.min()
        assert frame.max() <= 2*self.background_luminance, "frame maximum (%g) is greater than the maximum luminance (%g)" % (frame.max(), 2*self.background_luminance)
        return frame

    @property
    def img(self):
        r"""
        The current frame as a 2D array. A deferred frame is rendered whole on first access.
        """
        if self._img is None and self._deferred_frame is not None:
            self._img = self._check_luminance(self._render_frame(self._deferred_frame))
        return self._img

    @img.setter
    def img(self, frame):
        self._deferred_frame = None
        self._img = self._check_luminance(frame)

    def update(self):
        r"""
        Sets the current frame to the next frame in the sequence.
        """
        try:
            frame, self.variables = next(self._frames)
        except StopIteration:
            self.visible = False
        else:
            if isinstance(frame, numpy.ndarray):
                self.img = frame
            else:
                self._img = None
                self._deferred_frame = frame
            self._window_pixels = 0
        self._zoom_cache = OrderedDict()

    def reset(self):
        r"""
        Reset to the first frame in the sequence.
        """
        self.visible = True
        self._frames = self.frames()
        self.update()

    def next_frame(self):
        r"""For creating movies"""
        self.update()
        return [self.img]

    def frame_arrays(self):
        r"""
        Iterate over :func:`frames` with deferred frames rendered whole, i.e. as (2D array, variables) pairs.
        """
        for frame, variables in self.frames():
            if not isinstance(frame, numpy.ndarray):
                frame = self._render_frame(frame)
            yield frame, variables