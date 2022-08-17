import os
import time
import datetime
# import select
import json
from collections import OrderedDict
import subprocess as sp
import torch
from torch.autograd import Variable
from wavegan import *
from utils import parse_arguments
from logger import *
import numpy as np
from filelock import FileLock
from scipy.io import wavfile


STREAM_URL = 'url_from_youtube_or_twitch'
TRAJECTORY_CONFIG_PATH = os.path.join(os.getcwd(), 'traj.json')

SAMPLE_RATE = 16000
MAX_ARCHIVE_CHUNK = SAMPLE_RATE*2*60*60  # 1 hr of 8kHz 16-bit audio = ~57.6MB

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
LOGGER = logging.getLogger('wavegan')
LOGGER.setLevel(logging.DEBUG)

LOGGER.info('Initialized logger.')
init_console_logger(LOGGER)


def clean_state_dict(state_dict):
    state_dict_new = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        state_dict_new[name] = v
    return state_dict_new


def generate_trajectory(n_iter, dim, _z0=None, mov_last=None, jump=0.3, smooth=0.3, include_z0=True):
    _z = np.empty((n_iter + int(not include_z0), dim), dtype=np.float32)
    _z[0] = _z0 if _z0 is not None else np.random.random(dim) * 2 - 1
    mov = mov_last if mov_last is not None else (np.random.random(dim) * 2 - 1) * jump
    for i in range(1, len(_z)):
        mov = mov * smooth + (np.random.random(dim) * 2 - 1) * jump * (1 - smooth)
        mov -= (np.abs(_z[i-1] + mov) > 1) * 2 * mov
        _z[i] = _z[i-1] + mov
    return _z[-n_iter:], mov


def generate(model_dir, out_dir, ngpus):

    LOGGER.debug('generate_stream')

    cuda = torch.cuda.is_available()

    LOGGER.debug('loading model...')

    with FileLock('model'):
        netG, netD, latent_dim = load_model(model_dir, ngpus, cuda)

    LOGGER.debug('model loaded!')

    ###
    n_generate = 64
    seed1 = 562  # change this to change starting point
    np.random.seed(seed1)

    then = time.time()
    for idx in range(n_generate):
        z0 = np.random.random(latent_dim) * 2 - 1
        z = torch.from_numpy(z0.astype(np.float32))
        if cuda:
            z = z.cuda()

        sample_out = netG(Variable(z, requires_grad=False))
        if cuda:
            sample_out = sample_out.cpu()
        sample_out = sample_out.data.numpy()

        output_path = os.path.join(out_dir, "%02d.wav" % (idx))
        wavfile.write(output_path, SAMPLE_RATE, sample_out[0].T)
    now = time.time()
    elapsed = now - then
    print('took %f sec, (%f per sample)' % (elapsed, elapsed / n_generate))


def generate_traj(model_dir, out_dir, ngpus):

    LOGGER.debug('generate_stream')

    cuda = torch.cuda.is_available()

    LOGGER.debug('loading model...')

    with FileLock('model'):
        netG, netD, latent_dim = load_model(model_dir, ngpus, cuda)

    LOGGER.debug('model loaded!')

    ###
    # Move along local random trajectory in the latent space
    # (endless generative "improvisation" with smoothly changing breaks)
    n_generate = 64

    jump = 0.4  # factor of distance between adjacent trajectory points
    # (speed of changing)
    smooth = 0.25  # smoothing the trajectory turns, [0, 1]

    seed1 = 562  # change this to change starting point
    np.random.seed(seed1)
    z0 = np.random.random(latent_dim) * 2 - 1

    seed2 = 377  # change this to change trajectory
    np.random.seed(seed2)
    z, _ = generate_trajectory(n_generate, dim=latent_dim, _z0=z0, include_z0=True, jump=jump, smooth=smooth)
    z = torch.from_numpy(z)
    print(z.size())
    if cuda:
        z = z.cuda()

    sample_out = netG(Variable(z, requires_grad=False))
    if cuda:
        sample_out = sample_out.cpu()
    sample_out = sample_out.data.numpy()

    print(sample_out.shape)

    samples_unrolled = np.reshape(sample_out, (n_generate * sample_out.shape[2]))
    output_path = os.path.join(out_dir, "gen.wav")
    wavfile.write(output_path, SAMPLE_RATE, samples_unrolled)


def last_mod_time(model_dir):
    model_path = os.path.join(model_dir, "wavegan.pt")
    return os.path.getmtime(model_path)


def read_trajectory_params(json_path):
    with open(json_path, 'r') as json_file:
        config = json.load(json_file)
        return config['jump'], config['smooth']


def create_archive_pipe():
    archive_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    archive_path = os.path.join(os.getcwd(), 'archive/archive_%s.wav' % archive_id)
    command = ['ffmpeg',
               '-y',
               '-loglevel', 'error',
               '-f', 'f32le',
               '-acodec', 'pcm_f32le',
               '-ar', str(SAMPLE_RATE),
               '-ac', '1',
               '-i', 'pipe:0',
               # '-b:a', '128k',
               # '-c:a', 'libmp3lame',
               # '-reservoir', '0',
               # '-ar', '22050',
               archive_path
               ]
    archive_process = sp.Popen(command, stdin=sp.PIPE)
    archive_out = archive_process.stdin
    return archive_process, archive_out, archive_id


def load_model(model_dir, ngpus, cuda):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    model_size = config['model_size']
    latent_dim = config['latent_dim']
    netG = WaveGANGenerator(model_size=model_size, ngpus=ngpus, latent_dim=latent_dim, upsample=True)
    netD = WaveGANDiscriminator(model_size=model_size, ngpus=ngpus)

    if cuda:
        netG = netG.cuda()
        netD = netD.cuda()

    model_path = os.path.join(model_dir, "wavegan.pt")

    if cuda:
        checkpoint = torch.load(model_path)
        generator_state_dict = clean_state_dict(checkpoint["generator"])
        discriminator_state_dict = clean_state_dict(checkpoint["discriminator"])
    else:
        checkpoint = torch.load(model_path, map_location="cpu")
        generator_state_dict = checkpoint["generator"]
        discriminator_state_dict = checkpoint["discriminator"]
    netG.load_state_dict(generator_state_dict)
    netD.load_state_dict(discriminator_state_dict)

    return netG, netD, latent_dim


def generate_stream(model_dir, ngpus):
    LOGGER.debug('generate_stream')

    cuda = torch.cuda.is_available()

    LOGGER.debug('loading model...')

    with FileLock('model'):
        netG, netD, latent_dim = load_model(model_dir, ngpus, cuda)

    LOGGER.debug('model loaded!')

    ###
    # Move along local random trajectory in the latent space
    # (endless generative "improvisation" with smoothly changing breaks)
    n_generate = 1

    # factor of distance between adjacent trajectory points (speed of changing)
    jump = 0.25
    # smoothing the trajectory turns, [0, 1], close to 1 == straight lines
    smooth = 0.5
    if os.path.exists(TRAJECTORY_CONFIG_PATH):
        jump, smooth = read_trajectory_params(TRAJECTORY_CONFIG_PATH)
        last_traj_mod_time = os.path.getmtime(TRAJECTORY_CONFIG_PATH)
        LOGGER.info('loaded traj params: %f %f %f', jump, smooth, last_traj_mod_time)

    # change this to change starting point
    seed1 = 20
    # change this to change trajectory
    seed2 = 3

    last_model_mod_time = last_mod_time(model_dir)

    LOGGER.debug('getting ready...')

    # ffmpeg -y -loop 1 -i cover.png -f s16le -acodec pcm_s16le -ar 8000 -ac 1 -i pipe:0 -c:v libx264 -preset ultrafast -pix_fmt yuv420p -minrate 6000k -maxrate 6000k -bufsize 12000k -b:v 6000k -r 30 -g 30 -keyint_min 60 -x264opts "keyint=60:min-keyint=60:no-scenecut" -s 1920x1080 -tune zerolatency -b:a 128k -c:a aac -ar 48000 -strict experimental -f flv $STREAM_URL
    command = ['ffmpeg',
               '-y',
               '-loglevel', 'error',
               '-loop', '1',
               '-i', 'cover.png',
               '-f', 'f32le',
               '-acodec', 'pcm_f32le',
               '-ar', str(SAMPLE_RATE),
               '-ac', '1',
               '-i', 'pipe:0',
               '-c:v', 'libx264',
               '-preset', 'ultrafast',
               '-pix_fmt', 'yuv420p',
               '-minrate', '4500k',
               '-maxrate', '6000k',
               '-bufsize', '12000k',
               '-r', '30',
               '-q', '30',
               '-keyint_min', '60',
               '-x264opts', 'keyint=60:min-keyint=60:no-scenecut',
               '-s', '1920x1080',
               '-tune', 'zerolatency',
               '-b:a', '128k',
               '-c:a', 'aac',
               '-ar', '48000',
               '-strict', 'experimental',
               '-f', 'flv',
               STREAM_URL
               ]
    stream_pipe = sp.Popen(command, stdin=sp.PIPE)
    stream_out = stream_pipe.stdin

    archive_process, archive_out, archive_id = create_archive_pipe()
    archive_byte_count = 0

    z_archive = None

    LOGGER.debug('generating...')

    step_count = 0
    mov_last = None
    np.random.seed(seed1)
    z0 = np.random.random(latent_dim) * 2 - 1
    np.random.seed(seed2)
    while True:

        if step_count and (step_count % 38) == 0:  # ~5 minutes
            LOGGER.info('generated %d steps', step_count)

        model_mod_time = last_mod_time(model_dir)
        # print('%f %f' % (model_mod_time, last_model_mod_time))
        if model_mod_time > last_model_mod_time:
            # load new model
            with FileLock('model'):
                LOGGER.info('loading new model! %f', model_mod_time)
                netG, netD, _ = load_model(model_dir, ngpus, cuda)
                last_model_mod_time = model_mod_time

        if os.path.exists(TRAJECTORY_CONFIG_PATH):
            traj_mod_time = os.path.getmtime(TRAJECTORY_CONFIG_PATH)
            if traj_mod_time > last_traj_mod_time:
                jump, smooth = read_trajectory_params(TRAJECTORY_CONFIG_PATH)
                LOGGER.info('loaded new traj params: %f %f %f', jump, smooth, traj_mod_time)
                last_traj_mod_time = traj_mod_time

        z0, mov_last = generate_trajectory(n_generate, dim=latent_dim, _z0=z0, include_z0=False,
                                           mov_last=mov_last, jump=jump, smooth=smooth)
        z = torch.from_numpy(z0)

        if z_archive is not None:
            z_archive = torch.cat((z_archive, z))
        else:
            z_archive = z

        if cuda:
            z = z.cuda()

        sample_out = netG(Variable(z, requires_grad=False))
        if cuda:
            sample_out = sample_out.cpu()
        sample_out = sample_out.data.numpy()

        # print(sample_out.shape)

        samples_unrolled = np.reshape(sample_out, (n_generate * sample_out.shape[2]))
        samples_bytes = samples_unrolled.tobytes()

        stream_out.write(samples_bytes)
        archive_out.write(samples_bytes)

        step_count += 1

        archive_byte_count += len(samples_bytes)
        if archive_byte_count > MAX_ARCHIVE_CHUNK:
            torch.save(z_archive, 'archive/archive_%s.pt' % archive_id)
            z_archive = None
            archive_out.close()
            archive_process.terminate()
            archive_process, archive_out, archive_id = create_archive_pipe()
            archive_byte_count = 0

if __name__ == '__main__':
    LOGGER.debug('start')
    args = parse_arguments()
    # generate(args['model_dir'], args['output_dir'], args['ngpus'])