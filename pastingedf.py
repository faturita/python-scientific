import mne


# https://mne.tools/0.16/manual/io.html
# https://wonambi-python.github.io/analysis/tutorial.html#read-data


def raw_from_vwr(filename):

    #a = mne.io.read_raw_edf('/Users/rramele/Downloads/epilepsia/1/CL29082019F1.2.edf')
    from wonambi.ioeeg.micromed import Micromed

    m = Micromed(filename)

    #Open file with Xcode and press Command + Shift + J
    #Right click file name in left pane
    #Open as -> Hex


    #a = mne.io.read_raw_brainvision('/Users/rramele/Downloads/epilepsia/2/CL29082019F2.1.vhdr')


    [id, dat, freq, chn, samp, headers] = m.return_hdr()  


    ch_names = [c for c in chn]
    sfreq = freq
    data =m.return_dat(range(0,len(chn)),0,samp)

    ch_types = ['eeg' for _ in ch_names[0:-4]] + ['emg','emg','eog', 'eog'] 
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)
    #raw.add_events(events)

    #raw.plot_psd()

    #raw.filter(1,20)

    #raw.plot_psd()

    return raw

from mne.io import concatenate_raws, read_raw_edf

ruta = '/Users/rramele/Downloads/epilepsia/'

raw1 = raw_from_vwr(ruta+'/1/CL29082019F1.1.vwr')
raw2 = raw_from_vwr(ruta+'/1/CL29082019F1.2.vwr')
raw3 = raw_from_vwr(ruta+'/1/CL29082019F1.3.vwr')
raw4 = raw_from_vwr(ruta+'/1/CL29082019F1.4.vwr')
raw5 = raw_from_vwr(ruta+'/1/CL29082019F1.5.vwr')

raw = concatenate_raws([raw1,raw2,raw3, raw4, raw5])

from save_edf import write_edf

write_edf(raw, '1.edf', picks=None, tmin=0, tmax=None, overwrite=True)
