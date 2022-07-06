#!/usr/bin/env python3

import os
from util import *

from shutil import copyfile

from socket import gethostname

from datetime import datetime

from circular_buffer import *

import colorcet as cc
import hvplot.xarray  # noqa
import hvplot.pandas # noqa

import param, panel as pn

from joblib import delayed, Parallel
from functools import reduce


#pn.extension('terminal')
pn.extension('tabulator', template='material', sizing_mode='stretch_width')

basepath = os.path.split(__file__)[0]
confpath = pathjoin(basepath,'config')
with open(pathjoin(confpath,'settings.json')) as fid:
    config = json.load(fid)

username = os.environ.get('USER')
allusers = []
for p in config:
    if p.endswith('users'):
        allusers.extend(config[p])

userport = 5006
if username not in allusers:
   print(f'Error: user {username} not in list of valid users!')
   sys.exit(1)
userport += allusers.index(username)
        
scale = config['cmf_res']
year = config['cmf_year']
cmftype = config['cmf_types']
        
statepath = pathjoin(basepath,'output')
userpath = pathjoin(statepath,username)
donepath = pathjoin(userpath,'submitted')
if not os.path.exists(donepath):
    os.makedirs(donepath)
    
cssf = pathjoin(confpath,'style.css')
if pathexists(cssf):
    with open(cssf) as css:
        css = css.read()
    pn.extension(raw_css=[css])

wmts_urls = {
    "ESRI Imagery":"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg",
    "Digital Globe":"https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}",
    "OpenStreetMap":"https://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png",
    "Google Maps":"https://mt1.google.com/vt/x={X}&y={Y}&z={Z}",
    "Virtual Earth":"http://ecn.t3.tiles.virtualearth.net/tiles/a{Q}.jpeg?g=1",
    "SentinelHub":"https://services.sentinel-hub.com/ogc/wmts/{Q}"        
}
wmtsjsonf = pathjoin(confpath,'wmts.json')
if pathexists(wmtsjsonf):
    with open(wmtsjsonf) as fid:
        wmts_urls.update(json.load(fid))

wmts_default = "ESRI Imagery"
wmts_sites = list(wmts_urls.keys())
    
title = f'AVIRIS-NG CMF Tagger'
zoomcol = 'Zoom Level'
zpadcol = 'Zoom Buffer'

imgzoom = '[Fit Image]'
cidzoom = '[Fit Candidates]'
xzoom = '[Fit Width]'
yzoom = '[Fit Height]'
todohdr = '----------- ToDo Flightlines ------------'
donehdr = '----------- Done Flightlines ------------'

qchdrs = set([todohdr,donehdr,nonesel])

datafiles = []
lid2file = {}

state = {
    'initialized':False,
    'imagebounds':(None,None),
    'framebounds':(None,None),
    cmfcol:nonesel,
    zoomcol:nonesel,
    zpadcol:nonesel
}

# global state storage
data,plots,layout = None, None,None

renderer = hv.renderer('bokeh')
#renderer.webgl = True
#renderer.dpi = 10

frameheight = 700
framewidth  = 720

# cmdline parsing
parser = argparse.ArgumentParser('cmftagger.py')

# keyword arguments
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Enable verbose output')
parser.add_argument('--show', action='store_true',
                    help='Start browser on local machine to show running server')
parser.add_argument('--port', type=int, default=userport,
                    help='Manually specify server port')
parser.add_argument('--precache', action='store_true',
                    help='generate cached gtifs for all images in datapath a priori rather than on demand ')
parser.add_argument('--newassign', action='store_true',
                    help='generate new user assignments based on current plume list candidates')
parser.add_argument('--precachejobs', type=int, default=1,
                    help='number of parallel jobs to use in precaching operation')
parser.add_argument('--nouserfilt', action='store_true',
                    help='Do not filter flightlines by user assignments listed in qc spreadsheet')

parser.add_argument('--cmfcmap', type=str, default='YlOrRd',
                    help='CMF Colormap')

parser.add_argument('--viewerdims', type=int, nargs=2,
                    default=[frameheight,framewidth],
                    help='Image viewer dimensions [height width]')

args = parser.parse_args()

verbose = args.verbose
newassign = args.newassign
precache = args.precache
ncachejobs = args.precachejobs
userfilt = not args.nouserfilt
show = args.show
port = args.port
cmfcmap = args.cmfcmap
framedims = args.viewerdims

# zoom buffer radius in meters
zpaddef = '1km'
zpadopts = ['150m','1km','5km']
zpadvals = [int(s.replace('k','000')[:-1]) for s in zpadopts]
zpadvals = dict(zip(zpadopts,zpadvals))

# default slider settings
min_ppmm = 250
max_ppmm = 1500
min_prob = 50
max_prob = 100

# app pixel dimensions / aspect for scaling image view
frameheight,framewidth = framedims
frameaspect = frameheight / framewidth

xmargin  = lmargin = rmargin = 10 # x/left/right xmargins
ymargin  = tmargin = bmargin = 5  # y/top/bottom ymargins

titlemargin = (0,lmargin,0,rmargin)

titleheight = 30
layerheight = 31
lidheight = 95 # lidtable height
leftwidth = 335 # left panel width
pbarheight = 20 # progressbar height
cbarwidth = 90 # frame panel colorbar width
toolwidth = 40 # frame panel toolbar width
rowheight = 15 # tabulator row height 
logwidth  = 500 # header panel status/log message window width
msgheight = 150
logheight = msgheight + pbarheight + titleheight + tmargin + bmargin

appwidth = leftwidth+framewidth+cbarwidth+toolwidth+lmargin+rmargin


# note: use caution with data_aspect kwarg!!!
frameopts = dict(data_aspect=1.0,
                 frame_height=frameheight,frame_width=framewidth,
                 backend='bokeh',active_tools=['wheel_zoom'],
                 margin=(0,0,0,0))

pointsize = 20
pointcolor = {plumelab:'red',falselab:'cyan',amblab:'grey'}
pointopts = dict(size=pointsize,alpha=0.5,color=poscol,cmap=pointcolor)
labelcolor = {plumelab:'white',falselab:'black',amblab:'white'}
labelopts = dict(fontsize=int(round(pointsize*0.9)),
                 text_color=poscol,cmap=labelcolor,text_alpha=0.85,
                 padding=0.0,xoffset=0.0,yoffset=0.0)
    
opts.defaults(
    opts.Image(bgcolor='#999999',xaxis=None,yaxis=None),
    opts.Layout(merge_tools=True),
    opts.Points(**pointopts),
    opts.Labels(**labelopts)
)



# default paths


host = gethostname()
if host.startswith('hal9000'): # mbp laptop
    datapath = f'./store/ang/y{year[-2:]}/cmf/ch4/ort'
    title += ' @ hal9000.local'
elif host == 'dl': # uofa g3k
    datapath = f'/localstore/ang/y{year[-2:]}/cmf/ch4/ort'
    title += ' @ g3k.cminc.io'
else:
    datapath = f'/store/ang/y{year[-2:]}/cmf/ch4/ort'
    title += f' @ {host}'

gtifpath = pathjoin(datapath,'gtif')
if not pathexists(gtifpath):
    os.makedirs(gtifpath)

xlspath = pathjoin(basepath,'xls')
qcxlsf = os.path.join(xlspath,config['lidqc_template'])
plumexlsf = os.path.join(xlspath,config['plume_list'])
cnn_sheet,manualid_sheet = config['plume_sheets']
    
salpath = config.get('salimg_path',pathjoin(datapath,'plumes'))
labpath = config.get('labimg_path',pathjoin(datapath,'labels'))

data_match_pattern = f'ang{year}.*_({cmftype[0]}|{cmftype[1]})_v.*_img$' #f'ang{year}*_c*mf_v*_img'
salsuffix = '_prob_pos'
labsuffix = '_mask.png'

lid2userf = qcxlsf.replace('.xlsx','_lid2user.csv')
plumedf = load_plumedf(plumexlsf,cnn_sheet,manualid_sheet)
if newassign or not pathexists(lid2userf):
    print('Generating user->flightline assignments')
    lid2uidassign(plumedf,allusers,csvoutf=lid2userf)
    raw_input()

qcliddf,qclidnotes,lidvals,liddefs = collect_lidqcdat(qcxlsf,lid2userf,
                                                      username,donepath,
                                                      filter_uid=userfilt)

        
# progress / state + log messages widget
progresstitle = pn.pane.Markdown(f'**User {username}: 0 of {len(qcliddf)} flightlines complete**',
                                 sizing_mode='fixed',
                                 margin=titlemargin,height=titleheight)
progress = pn.indicators.Progress(name='Progress',
                                  margin=(0,lmargin,0,0),
                                  value=-1, max=len(qcliddf),
                                  bar_color='primary', active=True,
                                  sizing_mode='fixed',
                                  width=logwidth, height=pbarheight)

statebuf = CircularBuffer(max_size=50)
statemsg = pn.widgets.TextAreaInput(value='',
                                    margin=(tmargin,lmargin,0,0),
                                    width=logwidth,height=msgheight,
                                    placeholder='', disabled=True,
                                    sizing_mode='fixed',
                                    css_classes=['statemsg'])
#statemsg = pn.pane.HTML(object='',sizing_mode='stretch_height',width=logwidth)

# keep track of previous logtime to format log window msgs
logtime = None
def printstate(*args,style=None):
    global logtime,layout
    if statebuf.is_full():
        statebuf.dequeue()
    msg = ' '.join(args)
    curtime = f'{datestr(datetime.now(),fmt="%H:%M")}: '
    if curtime == logtime:
        curtime = ' '*len(curtime)
    else:
        logtime = curtime
    stampmsg = curtime+msg
    #if style is None:
    #    stampmsg = f'<p>{stampmsg}</p>'
    #else:
    #    stampmsg = f'<p style="{style}">{stampmsg}</p>'
        
    statebuf.enqueue(stampmsg)
    print(msg)
    statemsg.value = str(statebuf)

def update_datafiles():
    global datafiles, lid2file

    def next_datafile(qclist):
        # iterates over list of files in qclist to select next file
        # assumes 
        # - todo items precede done items in qclist
        # - qclist only contains valid filepaths and qchdr entries 
        nextfile = qclist[0]
        for qcfile in qclist:
            if qcfile not in qchdrs:
                nextfile = qcfile
                break
        return nextfile

    if len(datafiles)==0:
        datafiles = glob_re(data_match_pattern, os.listdir(datapath))
        datafiles = [pathsplit(cmff)[1] for cmff in datafiles]
        lid2file = dict([(cmff_to_lid(cmff),cmff) for cmff in datafiles])
        printstate(f'Found {len(datafiles)} total {cmftype} products')
    
    # updates cmfselector widget
    # - orders todo items first, done items last
    # - adds appropriate qchdr entries
    todomask = qcliddf[donecol]==0
    todolids = qcliddf.loc[todomask][lidcol].values
    todofiles = [lid2file[lid] for lid in todolids if lid in lid2file]

    donemask = qcliddf[donecol]==1
    donelids = qcliddf.loc[donemask,lidcol].values
    donefiles = [lid2file[lid] for lid in donelids if lid in lid2file]

    nfiles = len(donefiles)+len(todofiles)
    printstate(f'{username}: {len(donefiles)} of {nfiles} assigned flightlines complete')

    progress.max   = nfiles
    progress.value = len(donefiles) if len(donefiles)>0 else -1
    progresstitle.object = f'**User {username}: {len(donefiles)} of {progress.max} flightlines complete**'
    
    if nfiles==0:
        printstate(f'No flightlines available or none assigned to user {username}, exiting')
        sys.exit(1)

    sortedfiles = todofiles+donefiles
        
    if len(todofiles)==0:
        todofiles = [nonesel]
    if len(donefiles)==0:
        donefiles = [nonesel]

    # add todo + done files to cmfselector, pick next file
    cmfselector.options = [todohdr]+todofiles+[donehdr]+donefiles
    cmfselector.value = next_datafile(sortedfiles)

    return datafiles, lid2file

# widget initialization settings
apptitle = pn.pane.Markdown('**'+title+'**', height=titleheight,
                               margin=titlemargin)

cmfselector = pn.widgets.Select(name=f'CMF Image',
                                sizing_mode='stretch_width')

mapdef,mapopts = wmts_default, [nonesel]+wmts_sites
mapselector = pn.widgets.Select(name='Map Tiles',
                                value=mapdef,options=mapopts,
                                sizing_mode='stretch_width')

zoomdef,zoomopts = cidzoom,[cidzoom,imgzoom,xzoom,yzoom]
zoomselector = pn.widgets.Select(name=zoomcol,
                                 value=zoomdef,options=zoomopts,
                                 sizing_mode='stretch_width')

zpadselector = pn.widgets.Select(name=zpadcol,
                                 value=zpaddef,options=zpadopts,
                                 sizing_mode='stretch_width')
addcidhdr  = '----------- Select Label ------------'
addciddefs = [addcidhdr]+poslabs+neglabs+[amblab]
addcidtitle = pn.pane.Markdown('**Add User Candidate**', height=titleheight,
                            margin=titlemargin)
addcidselector = pn.widgets.Select(name='Label for New Candidate',
                                   value=addcidhdr,options=addciddefs,
                                   sizing_mode='stretch_width')
def addcidlabelcb(*events):
    # callback to add new user candidate to cidtable
    newlab = addcidselector.value
    if newlab == addcidhdr:
        return
    event = events[0]
    if event.type != 'changed':
        return
    printstate(f'Double click location for {newlab} candidate',
               style='color:red;')
    
addcidselector.param.watch(addcidlabelcb, ['value'], onlychanged=True)

delcidhdr  = '----------- Select Candidate ------------'
delciddefs = [delcidhdr,nonesel]
delcidtitle = pn.pane.Markdown('**Remove User Candidate**', height=titleheight,
                            margin=titlemargin)
delcidselector = pn.widgets.Select(name='User Candidate to Remove',
                                   value=delcidhdr,options=delciddefs,
                                   sizing_mode='stretch_width')
delcidbutton = pn.widgets.Button(name='Remove User Candidate', button_type='primary')
@pn.depends(delcidbutton,watch=True)
def delcidcb(pressed):
    # callback to update state for current lid
    # - submit / stash user qc state as completed 
    # - move current lid to done list
    # - update widgets and select next lid
    if not pressed:
        return

    delcid = delcidselector.value
    if delcid == nonesel:
        delcidselector.value = delcidhdr
    
    if delcidselector.value == delcidhdr:
        return

    cidtable.value = cidtable.value.loc[cidtable.value[cidcol]!=delcid].copy()
    data['candidates'] = cidtable.value.copy()

    zoomselector.options = zoomopts+[f'Candidate {cid}'
                                     for cid in cidtable.value[cidcol].values]
    if zoomselector.value == f'Candidate {delcid}':
        # if deleted candidate selectedl, reset to default zoom
        zoomselector.value = zoomdef

    delcidselector.value = delcidhdr
    delcidselector.options = [delcidhdr]+getusercids()
    
    printstate(f'Removed user candidate {delcid} ')

    # if 'candidates' in plots:
    #     dmap = plots['candidates']
    #     dmap.event(x=None,y=None,x_range=None,y_range=None)
    #     plot = renderer.get_plot(dmap)
    #     plot.refresh()
    
ppmmdefs = (min_ppmm,max_ppmm)
ppmmslider  = pn.widgets.IntRangeSlider(name='CMF Range (ppmm)',
                                        value=ppmmdefs,
                                        start=0, end=4000, step=25,                                        
                                        sizing_mode='stretch_width')
alphaslider  = pn.widgets.IntSlider(name='RGB Opacity (%)',
                                    value=100,
                                    start=0, end=100, step=5,                                        
                                    sizing_mode='stretch_width')
probdefs = (min_prob,max_prob)
probslider  = pn.widgets.IntRangeSlider(name='Salience Range (p(plume|cmf))',
                                        value=probdefs,
                                        start=0, end=100, step=5,
                                        sizing_mode='stretch_width')
    
# static label for the layerlbox since widget doesn't show title
layerlabel = pn.widgets.StaticText(value='Toggle Layers',height=20,
                                   sizing_mode='stretch_width',
                                   margin=(tmargin,lmargin,0,rmargin))
layeropts=['rgb', 'cmf', 'candidates', 'labels', 'salience']
layerdefs=['rgb', 'cmf', 'candidates']
layerboxes = pn.widgets.CheckButtonGroup(name='Layers', options=layeropts,
                                         value=layerdefs, 
                                         height=layerheight,
                                         sizing_mode='stretch_width',
                                         margin=(0,lmargin,bmargin,rmargin))
layerboxes.inline = True
def visible(layer):
    return layer in layerboxes.value

# qc table defaults
tabdefs = dict(
    theme='simple',
    configuration=dict(headerSort=False,resizableColumns=False),
    show_index=False,
    selectable=1,
    #row_height=rowheight,
)

# candidate qc table
cidzoomon = pn.widgets.Checkbox(name='Zoom to Selected Candidate', 
                                value=False, height=layerheight,
                                margin=(tmargin,lmargin,0,rmargin))
cidqclabs = poslabs+neglabs+amblabs
cidtitle = pn.pane.Markdown('**Candidate QC Table**',height=titleheight,
                            margin=titlemargin)
cidqccols = [cidcol,uidcol,latcol,loncol,xcol,ycol,labcol]
cidqcdefs = [None]*len(cidqccols)
cidformatters = {
    latcol: NumberFormatter(format='0[00].0[00000]'),
    loncol: NumberFormatter(format='0[00].0[00000]')
}
ciddf = pd.DataFrame([cidqcdefs],columns=cidqccols)
cideditors = {cidcol:None,latcol:None,loncol:None,
              labcol:SelectEditor(options=cidqclabs)}
cidtitles = {cidcol:'CID',latcol:'Latitude',loncol:'Longitude'}
cidtable = pn.widgets.Tabulator(value=ciddf,
                                editors=cideditors,
                                formatters=cidformatters,
                                sizing_mode='scale_width',
                                titles=cidtitles,
                                hidden_columns=[xcol,ycol,uidcol],
                                frozen_columns=[cidcol,latcol,loncol],
                                #height=rowheight*cidrows
                                row_height=rowheight,
                                layout='fit_data_stretch',
                                **tabdefs)

def cidtabzoomcb(*events):
    # callback to zoom to candidate selected in cidtable
    if cidzoomon.value == False:
        return
    
    zoomsel = state[zoomcol]
    if len(events[0].new)==1:
        cidindex = events[0].new[0]
        # zoom to candidate selected in table
        zoomsel = zoomselector.options[len(zoomopts)+cidindex]

        # update zoomselector (trigger render call) if selection changed
        if zoomsel != state[zoomcol]:
            zoomselector.value = zoomsel
            
cidtable.param.watch(cidtabzoomcb, ['selection'], onlychanged=True)    

# flightline qc table
lidtitle = pn.pane.Markdown('**Flightline QC Table**',height=titleheight,
                            margin=titlemargin)
lidqccols = [lidcol,uidcol]+list(liddefs.keys())
lidqcdefs = [None,username]+list(liddefs.values())
lideditors = dict([(col,SelectEditor(options=lidvals[col]))
                    for col in lidvals.keys()])
liddf = pd.DataFrame([lidqcdefs],columns=lidqccols)
lidtable = pn.widgets.Tabulator(value=liddf,                                
                                editors=lideditors,                                
                                hidden_columns=[lidcol,uidcol],
                                sizing_mode='scale_width',
                                layout='fit_columns',
                                # 2 rows=1 (header) + 1 (selected cmf)
                                #height=2*rowheight, 
                                **tabdefs)

submitbutton = pn.widgets.Button(name='Submit QC Data', button_type='primary')
@pn.depends(submitbutton,watch=True)
def submit(pressed):
    # callback to update state for current lid
    # - submit / stash user qc state as completed 
    # - move current lid to done list
    # - update widgets and select next lid
    if not pressed:
        return
    cmfimgf = state[cmfcol]
    cmflid = cmff_to_lid(cmfimgf)
    cidcsvf = cmfimgf+'_cid.csv'
    lidcsvf = cmfimgf+'_lid.csv'

    for csvf in (cidcsvf,lidcsvf):
        copyfile(pathjoin(userpath,csvf),
                 pathjoin(donepath,csvf))

    printstate(f'Submitted {pathsplit(cmfimgf)[1]}')
    qcliddf.loc[qcliddf[lidcol]==cmflid,donecol] = 1
    update_datafiles()

resetbutton = pn.widgets.Button(name='Reset QC Data', button_type='primary')
@pn.depends(resetbutton,watch=True)
def reset(pressed):
    # callback to reset state for current lid to values in plume list
    # - move current lid to todo list
    # - update widgets and select next lid
    if not pressed:
        return

    cmfimgf = state[cmfcol]
    cmflid = cmff_to_lid(cmfimgf)
    cidcsvf = cmfimgf+'_cid.csv'
    lidcsvf = cmfimgf+'_lid.csv'

    for csvf in (cidcsvf,lidcsvf):
        donef = pathjoin(donepath,csvf)
        if pathexists(donef):
            os.unlink(donef)

    data['candidates'] = data['plumelist'][cidqccols].copy()

    printstate(f'Reset {pathsplit(cmfimgf)[1]}')
    qcliddf.loc[qcliddf[lidcol]==cmflid,donecol] = 0
    update_datafiles()

    cidtable.value = data['candidates'][cidqccols].copy()

    layout[-1][-1].object = render(cmfselector.value,
                                   mapselector.value,
                                   ppmmslider.value,
                                   alphaslider.value,
                                   probslider.value,
                                   layerdefs,
                                   zoomdef,
                                   cidtable.value)

def getusercids():
    usercids = np.setdiff1d(cidtable.value[cidcol].values,
                            data['plumelist'][cidcol].values)
    if len(usercids)==0:
        return [nonesel]
    
    return list(usercids)
    
def cache_gtifs(cmffiles,ncores=4):
    # convert all cmffiles to cached geotiffs in parallel
    futures = [delayed(load_cache_gtif)(pathjoin(datapath,cmff),
                                        scale,gtifpath,njobs=1,
                                        cache_only=True)
               for cmff in cmffiles]
    # max out multiproc at 4 cores due to memory cap
    # NOTE (BDB, 05/10/21): memory errors occurred for me in dem generation step as well
    params = dict(n_jobs=ncores,verbose=10,max_nbytes=None)
    results = Parallel(**params)(futures)

    results = np.bool8(results)
    print(f'Caching succeeded for {np.count_nonzero(results)}/{len(results)} images.')
    if not results.all():
        failures = [pathsplit(cmff)[1] for cmfi,cmff in enumerate(cmffiles)
                    if results[cmfi]==False]
        print(f'caching failed for {len(failures)} images:'+', '.join(failures))

@pn.depends(lidtable,cidtable,watch=True)
def save_qcstate(liddf,ciddf,overwrite=True):
    # callback to update plume qc data (user csv files) in memory and on disk
    global state, plots, layout
    
    if state[cmfcol] in qchdrs:
        return
    if all(ciddf[cidcol].values==None):
        return

    cmfbase = pathsplit(state[cmfcol])[1]
    lidcsvf = pathjoin(userpath,cmfbase+'_lid.csv')
    if overwrite or not pathexists(lidcsvf):
        liddf.to_csv(lidcsvf,index=False)

    cidcsvf = pathjoin(userpath,cmfbase+'_cid.csv')
    if overwrite or not pathexists(cidcsvf):
        ciddf.to_csv(cidcsvf,index=False)

    if not state['initialized']:
        return
        
    cidlabs = ciddf[labcol].values
    datlabs = data['candidates'][labcol].values
    
    if len(cidlabs)==len(datlabs) and (cidlabs != datlabs).any():
        ciddiff = ciddf[cidcol].values[(cidlabs != datlabs)]
        print(f'Labels changed for CID(s) [{", ".join(ciddiff)}]')
        data['candidates'] = ciddf[cidqccols].copy()

    print(f'Cached {cmfbase} QC state')
    
    # if 'candidates' in plots:
    #     dmap = plots['candidates']
    #     plot = renderer.get_plot(dmap)
    #     dmap.event(x=None,y=None,x_range=None,y_range=None)        
    #     plot.refresh()

    # # redraw if plume labels changed
    # state[zoomcol] = nonesel
    # layout[-1][-1].object = render(cmfselector.value,
    #                                mapselector.value,
    #                                ppmmslider.value,
    #                                alphaslider.value,
    #                                probslider.value,
    #                                layerboxes.value,
    #                                zoomselector.value,cidtable.value)

def redraw():
    global layout
    layout[-1][-1].object = render(cmfselector.value,
                                   mapselector.value,
                                   ppmmslider.value,
                                   alphaslider.value,
                                   probslider.value,
                                   layerboxes.value,
                                   zoomselector.value,
                                   cidtable.value)

def load_qcstate():
    # restores previous state (csv files) for flightline/candidates per user
    liddf,ciddf = None,None
    cmfimgf = state[cmfcol]
    if cmfimgf in qchdrs:
        return liddf,ciddf
    
    cmfbase = pathsplit(state[cmfcol])[1]
    lidcsvf = pathjoin(userpath,cmfbase+'_lid.csv')
    cidcsvf = pathjoin(userpath,cmfbase+'_cid.csv')

    # if lid csv exists for current user, load it
    if pathexists(lidcsvf):
        liddf = pd.read_csv(lidcsvf)
        liddf.columns = lidqccols
    else:
        # otherwise use defaults
        liddf = pd.DataFrame([lidqcdefs],columns=lidqccols)
        liddf[lidcol] = [cmff_to_lid(cmfimgf)]
        liddf[uidcol] = [username]

    # if cid csv exists for current user, load it instead of using plume list
    if pathexists(cidcsvf):
        ciddf = pd.read_csv(cidcsvf)
        ciddf[cidcol] = [str(v) for v in ciddf[cidcol].values]
        ciddf[uidcol] = [username]*len(ciddf)
        ciddf.columns = cidqccols
    else:
        # otherwise get candidates from plume list
        ciddf = data['candidates'][cidqccols].copy()
    
    #printstate(f'Restored {cmfbase} qc state')
    return liddf, ciddf

def add_candidate(x,y,label):
    # make sure we didn't already add this candidate
    if cidexists(cidtable.value,x,y,label):
        return
    
    # create new candiate at location (x,y) with label newlab 
    lng,lat = meters_to_lnglat(x,y)
    
    newcid = next_cid(label,cidtable.value)
    newrow = {cidcol:newcid, labcol:label, xcol:x, ycol:y,
              loncol:lng, latcol:lat, uidcol:username}
    newdf = pd.DataFrame([[newrow[col] for col in cidqccols]],
                         columns=cidqccols)
    newdf = pd.concat([cidtable.value,newdf],axis=0)

    # update cidtable widget + candidate data
    cidtable.value = newdf.copy()
    data['candidates'] = newdf.copy()

    # add new cid to zoomselector + delcidselector
    zoomselector.options = zoomopts+[f'Candidate {cid}' for cid in newdf[cidcol].values]
    delcidselector.options = [delcidhdr]+getusercids()
    printstate(f'Added CID {newcid} at lon={lng:.3f}, lat={lat:3f}')

    # reset addcidselector to default header
    addcidselector.value = addcidhdr

def update_candidates(**kwargs):
    #print('kwargs: "%s"'%str((kwargs)))
    # value from cidtable.param.value
    # value = kwargs['value']
    
    # x_range,y_range from rangexy stream
    x_range,y_range = kwargs['x_range'],kwargs['y_range']

    # x,y from addxy stream
    newx,newy = kwargs['x'],kwargs['y']
    if None not in (newx,newy): 
        newlab = addcidselector.value
        if newlab == addcidhdr:
            print('No label selected for new candidates')
        else:
            add_candidate(newx,newy,newlab)

    return plot_candidates(cidtable.value,x_range,y_range)

def collect_data(cmfimgf):
    data = OrderedDict()
    layers = layerdefs.copy()
    
    cmfbase = cmfimgf if datapath not in cmfimgf else pathsplit(cmfimgf)[1]
    cmffile = cmfimgf if datapath in cmfimgf else pathjoin(datapath,cmfimgf)
    cmfdat = load_cache_gtif(cmffile,scale,gtifpath,unlock=True)

    cmfimg = cmfdat.sel(band=4)
    cmfimg.name = 'cmf'
    
    data['cmf'] = cmfimg.where(cmfimg.values > 0)
    
    rgbimg = cmfdat.sel(band=[1,2,3])

    #rgbimg = rgbimg.where((rgbimg.values!=0).all(axis=0),np.nan)
    #rgbimg = rgbimg.where((rgbimg.values==-9999).any(axis=0),np.nan)
    #rgbimg = (rgbimg-rdn_lim[0])/(rdn_lim[1]-rdn_lim[0])
    rgbimg.name = 'rgb'
    data['rgb'] = rgbimg

    # get xform matrix to convert (x,y) <-> (lng,lat)
    cmfxform = data['cmf'].attrs["transform"]
    cmfxform = rio.transform.Affine(*cmfxform)

    # plumelist = (read only) df of campaign plumes with uid assignments
    data['plumelist'] = cmf_plumes(plumedf,cmfimgf,cmfxform)
    data['plumelist'][uidcol] = [username]*len(data['plumelist'])
    
    # candidates = editable df containing campaign plumes (qc columns only)
    data['candidates'] = data['plumelist'][cidqccols].copy()
    
    labs = classlabs(data['candidates'][labcol].values)

    nplumes = np.count_nonzero(labs==1)

    labfile = pathjoin(labpath,cmfbase+labsuffix)
    if os.path.exists(labfile):
        printstate(f'{labfile} found, loading')
        if not labfile.endswith('.tif'):
            # convert png to gtif using cmf georeferencing metadata
            labgtif = labfile.replace('.png','.tif')            
            cmfmeta = rio.open(cmffile).meta.copy()
            labpng2tif(labfile,labgtif,cmfmeta)
            labfile = labgtif

        # load rgb label image
        labimg = load_cache_gtif(labfile,scale,gtifpath,resampling='max',
                                 outdtype='Byte',masked=True)

        # convert rgb label image to {-1,0,1} pixels
        r,g,b = [labimg.sel(band=i+1) for i in range(3)]
        labimg = (r/255.0) + -((g+b)/(2*255.0))
        # labeled pixels == 0 -> bg
        labimg.values[labimg.values==0] = np.nan
        # convert -1 -> 1
        labimg.values[labimg.values==-1] = 0
        labimg.name = 'labels'
        data['labels'] = labimg
        layers.append('labels')
        
    salfile = pathjoin(salpath,cmfbase+salsuffix)
    if os.path.exists(salfile):
        printstate(f'{salfile} found, loading')
        salimg = load_cache_gtif(salfile,scale,gtifpath)
        salimg = salimg.sel(band=1)
        # salience pixels == 0 -> bg
        salimg.values[salimg.values==0] = np.nan
        salimg.name = 'salience'
        data['salience'] = salimg
        layers.append('salience')
        
    layerboxes.options=layers
    layerboxes.value=[l for l in layers if l in layerboxes.value]

    return data

def init_plots(data):
    plots = OrderedDict()

    clipping_colors=dict(min='#00000000',NaN='#00000000')

    state['imagebounds'] = imagebounds(data['cmf'])
    state['framebounds'] = framebounds(data,frameaspect)
    
    plots['rgb'] = regrid(combine_bands(data['rgb'],rdn_lim))
    #plots['rgb'] = data['rgb'].hvplot.rgb(rasterize=True,
    #                                      attr_labels=True,
    #                                      tools=rgbtools)
    
    cmftools = hovertool('cmf')
    plots['cmf'] = data['cmf'].hvplot.image(z='cmf',
                                            rasterize=True,
                                            cmap=cmfcmap,
                                            color='cmf',
                                            clabel='ppmm',
                                            attr_labels=True,
                                            tools=cmftools)
    plots['cmf'].opts(clipping_colors=clipping_colors)

    if 'labels' in data:
        plots['labels'] = data['labels'].hvplot.image(z='labels',
                                                      rasterize=True,
                                                      cmap='hsv',
                                                      color='labels',
                                                      clim=(-1, 1),
                                                      colorbar=False,
                                                      attr_labels=True)
    if 'salience' in data:
        plots['salience'] = data['salience'].hvplot.image(z='salience',
                                                          rasterize=True,
                                                          cmap='RdBu_r',
                                                          color='salience',
                                                          clim=(0, 1),
                                                          colorbar=False,                                                          
                                                          attr_labels=True)                                                          
        plots['salience'].opts(clipping_colors=clipping_colors)

    addxy = hv.streams.DoubleTap(source=plots['cmf'])

    #plots['candidates'] = plot_candidates(data['candidates'])

    #pointdraw = streams.PointDraw(data=plots['candidates'].columns(),
    #                              num_objects=len(data['candidates']),
    #                              source=plots['candidates'],
    #                              empty_value='black')    

    # add rangexy stream to render candidate bounds
    cid_streams = []
    for stream in plots['cmf'].streams:
        if isinstance(stream,hv.streams.RangeXY):
            cid_streams = [stream]
            break

    assert(len(cid_streams)==1)
    cid_streams.append(addxy)
    cid_streams.append(cidtable.param.value)

    plots['candidates'] = hv.DynamicMap(update_candidates,streams=cid_streams)
    plots['candidates'].opts(opts.Points(tools=hovertool('candidates')))

    plots['composite'] = reduce(lambda p, pi: p*pi, plots.values())
    
    return plots
                                                           
@pn.depends(cmfselector,mapselector,ppmmslider,alphaslider,probslider,
            layerboxes,zoomselector,zpadselector,cidtable)
def render(selcmf,tilesrc,ppmmrange,rgbalpha,probrange,layers,
           selzoom,selzpad,ciddf):
    global state,data,plots,layout
    plotkw = frameopts.copy()
    pointkw = {}
    
    if selcmf in qchdrs:
        # header selected in cmfselector, replace w/ existing state value
        cmfselector.value =  state[cmfcol]
    elif selcmf != state[cmfcol]:
        # new image selected, load it
        printstate(f'Loading {pathsplit(selcmf)[1]}')
        
        state['initialized'] = False
        if layout is not None:
            imgview = layout[-1][-1]
            for elt in (imgview,cidtable,lidtable):
                elt.loading = True

        state[cmfcol] = selcmf
        data = collect_data(selcmf)
        plots = init_plots(data)

        liddf, ciddf = load_qcstate()

        if liddf is not None:
            lidtable.value = liddf[lidqccols].copy()

        if ciddf is not None:
            cidtable.value = ciddf[cidqccols].copy()
            cidtable.selection = []

        # trigger a zoom operation 
        state[zoomcol] = state[zpadcol] = nonesel
        zoomselector.value = zoomdef
        zoomselector.options = zoomopts+[f'Candidate {cid}'
                                         for cid in cidtable.value[cidcol].values]

        # update list of user cids
        delcidselector.value = delcidhdr
        delcidselector.options = [delcidhdr]+getusercids()

    # update table height in case contents changed
    cidtable.height = rowheight*len(cidtable.value)
    lidtable.height = rowheight*2 
    
    if selzoom != state[zoomcol] or selzpad != state[zpadcol]:
        state[zoomcol] = selzoom
        state[zpadcol] = selzpad
        zoompad = zpadvals[selzpad]
        if selzoom not in zoomopts:
            # specific cid selected
            cid = selzoom.split()[-1]
            cidmask = cidtable.value[cidcol]==cid
            pxy = cidtable.value[cidmask][[xcol,ycol]].values[0]
            plotkw['xlim'] = (pxy[0]-zoompad,pxy[0]+zoompad)
            plotkw['ylim'] = (pxy[1]-zoompad,pxy[1]+zoompad)
            tabcids = np.array(cidtable.value[cidcol].values,dtype=str)
            cidtable.selection = [np.argmax(tabcids==str(cid))]
        elif selzoom == cidzoom:
            # zoom to bbox surrounding all cids
            px = cidtable.value[xcol].values
            py = cidtable.value[ycol].values
            plotkw['xlim'] = (px.min()-zoompad,px.max()+zoompad)
            plotkw['ylim'] = (py.min()-zoompad,py.max()+zoompad)
            cidtable.selection = [] # deselect current cid 
        elif selzoom == imgzoom:
            xlim,ylim = state['imagebounds']
            plotkw['xlim'] = xlim
            plotkw['ylim'] = ylim
            cidtable.selection = [] # deselect current cid
        elif selzoom in (xzoom,yzoom):
            xlim,ylim = state['imagebounds']
            xrad,yrad = (xlim[1]-xlim[0])/2, (ylim[1]-ylim[0])/2
            if selzoom==xzoom:
                ymid = ylim[0]+yrad
                ylim = (ymid-xrad,ymid+xrad)
            elif selzoom==yzoom:
                xmid = xlim[0]+xrad
                xlim = (xmid-yrad,xmid+yrad)
            plotkw['xlim'] = xlim
            plotkw['ylim'] = ylim
            
            cidtable.selection = [] # deselect current cid            
            
    else:
        plot = renderer.get_plot(plots['rgb']).state
        plotkw['xlim'] = (plot.x_range.start,plot.x_range.end)
        plotkw['ylim'] = (plot.y_range.start,plot.y_range.end)

    if 'xlim' in plotkw and 'ylim' in plotkw:
         pointkw['xlim']=plotkw['xlim']
         pointkw['ylim']=plotkw['ylim']

    rgbalpha = alphaslider.value_throttled/100.0 if visible('rgb') else 0.0         
    plots['rgb'].opts(alpha=rgbalpha)

    ppmmlim = ppmmslider.value_throttled
    #ppmmslider.visible = visible('cmf')
    plots['cmf'].opts(clim=ppmmlim,alpha=float(visible('cmf')))
    
    if 'salience' in plots:
        #probslider.visible = visible('salience')
        problim = (probslider.value_throttled[0]/100.0,
                   probslider.value_throttled[1]/100.0)
        plots['salience'].opts(clim=problim,alpha=float(visible('salience')))
    else:
        probslider.visible = False

    if 'labels' in plots:
        plots['labels'].opts(alpha=float(visible('labels')))

    if 'candidates' in plots:
        plots['candidates'].opts(
            opts.Points(alpha=float(visible('candidates'))),
            opts.Labels(text_alpha=float(visible('candidates')))
        )

    if layout is not None:
        imgview = layout[-1][-1]
        for elt in (imgview,cidtable,lidtable):
            elt.loading = False

    state['initialized'] = True

    def format_figure(plot, element):
        p = plot.state
        p.toolbar.logo = None
        p.tools = [t for t in p.tools if not isinstance(t,ResetTool)]        
        if len(p.legend)==0:
            return # plot not initialized yet
        p.legend.background_fill_color = "white"
        p.legend.background_fill_alpha = 0.5
        p.legend.border_line_width = 1
        p.legend.label_text_color = "black"
            
        #print('plotkw: "%s"'%str((plotkw)))
        #print('update toolbar: dir(p):\n%s'%str((dir(p))))
        #print('x_range, y_range: "%s"'%str(((p.x_range.start, p.x_range.end),
        #                                    (p.y_range.start, p.y_range.end))))

    plotkw['hooks'] = [format_figure]

    plots['composite'].opts(opts.Image(**plotkw),
                            opts.Points(**pointkw),
                            opts.Labels(**pointkw))

    if tilesrc != nonesel:
        return hv.Tiles(wmts_urls[tilesrc],name=tilesrc) * plots['composite']
    
    return plots['composite']

datafiles, lid2file = update_datafiles()

if precache:
    print(f'starting precache processing for {len(datafiles)} image files.')
    cache_gtifs(datafiles,ncores=ncachejobs)
    print(f'precache processing complete.')

# # hack to avoid weird image rescaling issues in initial render calls
# outfig = render(cmfselector.value,mapselector.value,ppmmslider.value,
#                 alphaslider.value,probslider.value,layerdefs,zoomdef,
#                 cidtable.value)
# state[zoomcol] = nonesel

layout = pn.Column(
    pn.Row(
        pn.WidgetBox(
            pn.Column(apptitle,
                      pn.Row(cmfselector,pn.Row(zoomselector,zpadselector)),
                      pn.Row(mapselector,pn.Column(layerlabel,
                                                   layerboxes)),
                      pn.Row(ppmmslider,alphaslider,probslider)
            )
        ),
        pn.Column(progresstitle,progress,statemsg,
                  sizing_mode='fixed',width=logwidth,height=logheight),
        sizing_mode='stretch_height', width=appwidth
    ),
    pn.Row(
        pn.WidgetBox(
            pn.Column(lidtitle,lidtable),
            sizing_mode='stretch_width',
            height=lidheight,
        ),
        sizing_mode='stretch_height', width=appwidth
    ),
    pn.Row(
        pn.Column(
            pn.WidgetBox(pn.Column(cidtitle,cidtable,cidzoomon)),
            pn.WidgetBox(pn.Column(addcidtitle,addcidselector)),
            pn.WidgetBox(pn.Column(delcidtitle,delcidselector,delcidbutton)),            
            pn.WidgetBox(pn.Column(submitbutton,resetbutton)),
            sizing_mode='stretch_height',width=leftwidth,
        ),
        pn.Column(render),
        sizing_mode='stretch_height', width=appwidth
    ),
    #sizing_mode='fixed', width=appwidth
)

#layout.show(title=apptitle,threaded=True)
pn.serve(layout,title=title,start=True,show=show,port=port,threaded=False)
