# importing libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import glob
import sys
import argparse

def fit_sin(tt, yy, blinking_freq=1000, min_freq=3.0, max_frame_time_ms=100):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    # https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))

    min_freq_idx = np.argmax(ff > 3.0) # assume at least 3 Hz
    # Limit maximum frequency based on the fps - frame readout cannot be more than frame delay time
    max_freq_index = np.argmax(ff > max_frame_time_ms * blinking_freq/1000)
    Fyy[max_freq_index:-max_freq_index] *= 0
    guess_freq = abs(ff[np.argmax(Fyy[min_freq_idx:-min_freq_idx])+min_freq_idx])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}

def estimate_rs(videofile, blinking_freq=1000, start_time = 1, samples=50, show_plot=True):
    print(f"Analyzing {videofile}")
    vidavg = None
    debias_frames = samples
    sample_frames = samples
    start_time = round(1000*start_time)
    show_graph = show_plot

    framecount = 0
    cap = cv2.VideoCapture(videofile)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_MSEC,start_time)
    ret, frame = cap.read()


    result = {
        "frame_readout_time": None,
        "frame_readout_times": [],
        "error_estimate": None,
        "filename": videofile,
        "blinking_freq": blinking_freq,
        "width": width,
        "height": height,
        "fps": fps,
        "fraction_good": None,
        "success": False,
        "error": True
    }

    # Check if opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video file")
        return result
       
    # Read until video is completed
    print(f"Computing average of {debias_frames} frames")
    while(cap.isOpened()):
          
      # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            # reduce to one channel, can be either greyscale or specific color
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #frame[:,:,0] # blue green red #
            vidavg = grey.astype("float64") if type(vidavg) == type(None) else vidavg + grey.astype("float64")
            framecount += 1
            #print("Reading frame {}".format(framecount))

            if framecount == debias_frames or debias_frames == 0:
              break

        else: 
            break

    vidavg = (vidavg/framecount).astype("int16")
    print("Average computed")

    framecount = 0
    est_times = []
    cap.set(cv2.CAP_PROP_POS_MSEC,start_time)
    while(cap.isOpened()):
          
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            hcomb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hcomb = hcomb - vidavg if debias_frames != 0 else hcomb #= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#[:,0:120]
            # Display the resulting frame
            #cv2.imshow('Frame', np.clip(hcomb+128, 0, 255).astype("uint8"))
            vavg = np.mean(hcomb, axis=1)
            n = vavg.shape[0]
            t = np.linspace(0,1,vavg.shape[0])
            #plt.plot(t,vavg)
            #plt.show()
            #sp = np.fft.fft(vavg)
            #freq = np.fft.fftfreq(t.shape[-1],d=1/vavg.shape[0])
            #plt.plot(freq[2:n//2]/blinking_freq * 1000, np.abs(sp[2:n//2]))
            #plt.xlabel("RS time")
            #plt.show()

            autofit = fit_sin(t, vavg, blinking_freq, min_freq=3.0, max_frame_time_ms=1000/fps)
            #print(autofit)

            detectedrate = autofit["omega"]/(2*np.pi) # Hz
            rs_time = detectedrate/blinking_freq * 1000
            est_times.append(rs_time)
            print(f"RS estimate frame {framecount}: {rs_time}")
            if show_graph:
                fig, axs = plt.subplots(2)
                ax1 = plt.subplot(211)
                ax1.plot(t, vavg, '.')
                #plt.plot(t, data_first_guess, label='first guess')
                ax1.plot(t, autofit["amp"]*np.sin(autofit["omega"]*t+autofit["phase"])+autofit["offset"])
                ax1.set_xlim([0, 1])
                plt.title(f"Estimated RS: {rs_time:0.3f} ms for first analyzed frame of {videofile}")
                plt.ylabel("Intensity")
                #axs[1].ylabel("Color intensity")
                #plt.plot(fine_t, data_fit, label='after fitting')
                #plt.legend()
                ax2 = plt.subplot(212)
                preview_height = 400
                ax2.pcolormesh(np.linspace(0,height,int(height)),np.linspace(0,width,preview_height),cv2.resize(hcomb, (preview_height, int(height))).transpose())
                plt.ylabel("Frame width")
                plt.xlabel("Frame height")
                show_graph = False

            framecount += 1
            if framecount == sample_frames:
                break

        else: 
            break
       
    cap.release()
       
    # Closes all the frames
    cv2.destroyAllWindows()
    est_times = np.array(est_times)
    median = np.median(est_times)

    outlier_threshold = 0.2
    no_outliers = est_times[np.abs(est_times - median) < outlier_threshold]
    frame_readout_time = np.mean(no_outliers)
    error = max((frame_readout_time - no_outliers).max(), abs((frame_readout_time - no_outliers).min()))
    fraction_good = no_outliers.shape[0]/sample_frames
    success = fraction_good > 0.7 # 70% of frames agree


    print(f"Number of frames after removing outliers: {no_outliers.shape[0]} out of {sample_frames} = {fraction_good*100:0.1f}%")
    print(f"----\nFinal rolling shutter estimate [ms]: {frame_readout_time:0.6f} +/- {error:0.6f}\n")
    plt.show()

    # If less than this, assume bad calibration and compute error including outliers.
    if not success:
        error = max((frame_readout_time - est_times).max(), abs((frame_readout_time - est_times).min()))
    
    result.update({
        "frame_readout_time": frame_readout_time,
        "frame_readout_times": list(est_times),
        "error_estimate": error,
        "fraction_good": fraction_good,
        "success": success,
        "error": False
    })

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Video file(s), supports wildcard e.g. examples\\*", nargs=1)
    parser.add_argument('--freq', '-f', help="LED blinking frequency in Hz. Default=1000 Hz", type= float, default=1000)
    parser.add_argument('--start', '-s', help="Video timestamp to begin analysis in seconds, default=1 s", type=float, default=1)
    parser.add_argument('--samples', '-n', help="Number of frames to analyze, default=50", type=int, default=50)
    parser.add_argument('--plot', help="Show plot of the analysis of the first frame", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    files = args.file[0]

    result = []

    for name in glob.glob(files):
        result.append(estimate_rs(name, blinking_freq=args.freq, start_time = args.start, samples=args.samples, show_plot=args.plot))
    print(f"Analyzed with blinking frequency of {args.freq}.")
    print("--- RESULTS ---")
    print("filename, format, success, frame_readout_time[ms] ± error[ms]")
    for res in result:
        print(f"{res['filename']}, {res['width']:.0f}x{res['height']:.0f}@{res['fps']:.2f}, {res['success']}, {res['frame_readout_time']:0.3f} ± {res['error_estimate']:0.5f}")

