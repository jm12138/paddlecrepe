import warnings

import numpy as np
import resampy
import paddle
import tqdm

import paddlecrepe


__all__ = ['CENTS_PER_BIN',
           'MAX_FMAX',
           'PITCH_BINS',
           'SAMPLE_RATE',
           'WINDOW_SIZE',
           'UNVOICED',
           'embed',
           'embed_from_file',
           'embed_from_file_to_file',
           'embed_from_files_to_files',
           'infer',
           'predict',
           'predict_from_file',
           'predict_from_file_to_file',
           'predict_from_files_to_files',
           'preprocess',
           'postprocess',
           'resample']


###############################################################################
# Constants
###############################################################################


CENTS_PER_BIN = 20  # cents
MAX_FMAX = 2006.  # hz
PITCH_BINS = 360
SAMPLE_RATE = 16000  # hz
WINDOW_SIZE = 1024  # samples
UNVOICED = np.nan


###############################################################################
# Crepe pitch prediction
###############################################################################


def predict(audio,
            sample_rate,
            hop_length=None,
            fmin=50.,
            fmax=MAX_FMAX,
            model='full',
            decoder=paddlecrepe.decode.viterbi,
            return_harmonicity=False,
            return_periodicity=False,
            batch_size=None,
            pad=True):
    """Performs pitch estimation

    Arguments
        audio (paddle.tensor [shape=(1, time)])
            The audio signal
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_harmonicity (bool) [DEPRECATED]
            Whether to also return the network confidence
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (paddle.tensor [shape=(1, 1 + int(time // hop_length))])
        (Optional) periodicity (paddle.tensor
                                [shape=(1, 1 + int(time // hop_length))])
    """
    # Deprecate return_harmonicity
    if return_harmonicity:
        message = (
            'The paddlecrepe return_harmonicity argument is deprecated and '
            'will be removed in a future release. Please use '
            'return_periodicity. Rationale: if network confidence measured '
            'harmonics, the value would be low for non-harmonic, periodic '
            'sounds (e.g., sine waves). But this is not observed.')
        warnings.warn(message, DeprecationWarning)
        return_periodicity = return_harmonicity

    results = []

    # Postprocessing breaks gradients, so just don't compute them
    with paddle.no_grad():

        # Preprocess audio
        generator = preprocess(audio,
                               sample_rate,
                               hop_length,
                               batch_size,
                               pad)
        for frames in generator:

            # Infer independent probabilities for each pitch bin
            probabilities = infer(frames, model)

            # shape=(batch, 360, time / hop_length)
            probabilities = probabilities.reshape(
                audio.shape[0], -1, PITCH_BINS).transpose(1, 2)

            # Convert probabilities to F0 and periodicity
            result = postprocess(probabilities,
                                 fmin,
                                 fmax,
                                 decoder,
                                 return_periodicity)

            if isinstance(result, tuple):
                result = (result[0],
                          result[1])
            else:
                 result = result

            results.append(result)

    # Split pitch and periodicity
    if return_periodicity:
        pitch, periodicity = zip(*results)
        return paddle.concat(pitch, 1), paddle.concat(periodicity, 1)

    # Concatenate
    return paddle.concat(results, 1)


def predict_from_file(audio_file,
                      hop_length=None,
                      fmin=50.,
                      fmax=MAX_FMAX,
                      model='full',
                      decoder=paddlecrepe.decode.viterbi,
                      return_harmonicity=False,
                      return_periodicity=False,
                      batch_size=None,
                      pad=True):
    """Performs pitch estimation from file on disk

    Arguments
        audio_file (string)
            The file to perform pitch tracking on
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        return_harmonicity (bool) [DEPRECATED]
            Whether to also return the network confidence
        return_periodicity (bool)
            Whether to also return the network confidence
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        pitch (paddle.tensor [shape=(1, 1 + int(time // hop_length))])
        (Optional) periodicity (paddle.tensor
                                [shape=(1, 1 + int(time // hop_length))])
    """
    # Load audio
    audio, sample_rate = paddlecrepe.load.audio(audio_file)

    # Predict
    return predict(audio,
                   sample_rate,
                   hop_length,
                   fmin,
                   fmax,
                   model,
                   decoder,
                   return_harmonicity,
                   return_periodicity,
                   batch_size,
                   pad)


def predict_from_file_to_file(audio_file,
                              output_pitch_file,
                              output_harmonicity_file=None,
                              output_periodicity_file=None,
                              hop_length=None,
                              fmin=50.,
                              fmax=MAX_FMAX,
                              model='full',
                              decoder=paddlecrepe.decode.viterbi,
                              batch_size=None,
                              pad=True):
    """Performs pitch estimation from file on disk

    Arguments
        audio_file (string)
            The file to perform pitch tracking on
        output_pitch_file (string)
            The file to save predicted pitch
        output_harmonicity_file (string or None) [DEPRECATED]
            The file to save predicted harmonicity
        output_periodicity_file (string or None)
            The file to save predicted periodicity
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio
    """
    # Deprecate output_harmonicity_file
    if output_harmonicity_file is not None:
        message = (
            'The paddlecrepe output_harmonicity_file argument is deprecated and '
            'will be removed in a future release. Please use '
            'output_periodicity_file. Rationale: if network confidence measured '
            'harmonic content, the value would be low for non-harmonic, periodic '
            'sounds (e.g., sine waves). But this is not observed.')
        warnings.warn(message, DeprecationWarning)
        output_periodicity_file = output_harmonicity_file

    # Predict from file
    prediction = predict_from_file(audio_file,
                                   hop_length,
                                   fmin,
                                   fmax,
                                   model,
                                   decoder,
                                   False,
                                   output_periodicity_file is not None,
                                   batch_size,
                                   pad)

    # Save to disk
    if output_periodicity_file is not None:
        paddle.save(prediction[0].detach(), output_pitch_file)
        paddle.save(prediction[1].detach(), output_periodicity_file)
    else:
        paddle.save(prediction.detach(), output_pitch_file)


def predict_from_files_to_files(audio_files,
                                output_pitch_files,
                                output_harmonicity_files=None,
                                output_periodicity_files=None,
                                hop_length=None,
                                fmin=50.,
                                fmax=MAX_FMAX,
                                model='full',
                                decoder=paddlecrepe.decode.viterbi,
                                batch_size=None,
                                pad=True):
    """Performs pitch estimation from files on disk without reloading model

    Arguments
        audio_files (list[string])
            The files to perform pitch tracking on
        output_pitch_files (list[string])
            The files to save predicted pitch
        output_harmonicity_files (list[string] or None) [DEPRECATED]
            The files to save predicted harmonicity
        output_periodicity_files (list[string] or None)
            The files to save predicted periodicity
        hop_length (int)
            The hop_length in samples
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        decoder (function)
            The decoder to use. See decode.py for decoders.
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio
    """
    # Deprecate output_harmonicity_files
    if output_harmonicity_files is not None:
        message = (
            'The paddlecrepe output_harmonicity_files argument is deprecated and '
            'will be removed in a future release. Please use '
            'output_periodicity_files. Rationale: if network confidence measured '
            'harmonic content, the value would be low for non-harmonic, periodic '
            'sounds (e.g., sine waves). But this is not observed.')
        warnings.warn(message, DeprecationWarning)
        output_periodicity_files = output_harmonicity_files

    if output_periodicity_files is None:
        output_periodicity_files = len(audio_files) * [None]

    # Setup iterator
    iterator = zip(audio_files, output_pitch_files, output_periodicity_files)
    iterator = tqdm.tqdm(iterator, desc='paddlecrepe', dynamic_ncols=True)
    for audio_file, output_pitch_file, output_periodicity_file in iterator:

        # Predict a file
        predict_from_file_to_file(audio_file,
                                  output_pitch_file,
                                  None,
                                  output_periodicity_file,
                                  hop_length,
                                  fmin,
                                  fmax,
                                  model,
                                  decoder,
                                  batch_size,
                                  pad)

###############################################################################
# Crepe pitch embedding
###############################################################################


def embed(audio,
          sample_rate,
          hop_length=None,
          model='full',
          batch_size=None,
          pad=True):
    """Embeds audio to the output of CREPE's fifth maxpool layer

    Arguments
        audio (paddle.tensor [shape=(1, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        embedding (paddle.tensor [shape=(1,
                                        1 + int(time // hop_length), 32, -1)])
    """
    results = []

    # Preprocess audio
    generator = preprocess(audio,
                           sample_rate,
                           hop_length,
                           batch_size,
                           pad)
    for frames in generator:

        # Infer pitch embeddings
        embedding = infer(frames, model, embed=True)

        # shape=(batch, time / hop_length, 32, embedding_size)
        result = embedding.reshape((audio.shape[0], frames.shape[0], 32, -1))

        results.append(result)

    # Concatenate
    return paddle.concat(results, 1)


def embed_from_file(audio_file,
                    hop_length=None,
                    model='full',
                    batch_size=None,
                    pad=True):
    """Embeds audio from disk to the output of CREPE's fifth maxpool layer

    Arguments
        audio_file (string)
            The wav file containing the audio to embed
        hop_length (int)
            The hop_length in samples
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        embedding (paddle.tensor [shape=(1,
                                        1 + int(time // hop_length), 32, -1)])
    """
    # Load audio
    audio, sample_rate = paddlecrepe.load.audio(audio_file)

    # Embed
    return embed(audio,
                 sample_rate,
                 hop_length,
                 model,
                 batch_size,
                 pad)


def embed_from_file_to_file(audio_file,
                            output_file,
                            hop_length=None,
                            model='full',
                            batch_size=None,
                            pad=True):
    """Embeds audio from disk and saves to disk

    Arguments
        audio_file (string)
            The wav file containing the audio to embed
        hop_length (int)
            The hop_length in samples
        output_file (string)
            The file to save the embedding
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio
    """
    # No use computing gradients if we're just saving to file
    with paddle.no_grad():

        # Embed
        embedding = embed_from_file(audio_file,
                                    hop_length,
                                    model,
                                    batch_size,
                                    pad)

        # Save to disk
        paddle.save(embedding.detach(), output_file)


def embed_from_files_to_files(audio_files,
                              output_files,
                              hop_length=None,
                              model='full',
                              batch_size=None,
                              pad=True):
    """Embeds audio from disk and saves to disk without reloading model

    Arguments
        audio_files (list[string])
            The wav files containing the audio to embed
        output_files (list[string])
            The files to save the embeddings
        hop_length (int)
            The hop_length in samples
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio
    """
    # Setup iterator
    iterator = zip(audio_files, output_files)
    iterator = tqdm.tqdm(iterator, desc='paddlecrepe', dynamic_ncols=True)
    for audio_file, output_file in iterator:

        # Embed a file
        embed_from_file_to_file(audio_file,
                                output_file,
                                hop_length,
                                model,
                                batch_size,
                                pad)


###############################################################################
# Components for step-by-step prediction
###############################################################################


def infer(frames, model='full', embed=False):
    """Forward pass through the model

    Arguments
        frames (paddle.tensor [shape=(time / hop_length, 1024)])
            The network input
        model (string)
            The model capacity. One of 'full' or 'tiny'.
        embed (bool)
            Whether to stop inference at the intermediate embedding layer

    Returns
        logits (paddle.tensor [shape=(1 + int(time // hop_length), 360)]) OR
        embedding (paddle.tensor [shape=(1 + int(time // hop_length),
                                       embedding_size)])
    """
    # Load the model if necessary
    if not hasattr(infer, 'model') or not hasattr(infer, 'capacity') or \
       (hasattr(infer, 'capacity') and infer.capacity != model):
        paddlecrepe.load.model(model)

    infer.model = infer.model

    # Apply model
    return infer.model(frames, embed=embed)


def postprocess(probabilities,
                fmin=0.,
                fmax=MAX_FMAX,
                decoder=paddlecrepe.decode.viterbi,
                return_harmonicity=False,
                return_periodicity=False):
    """Convert model output to F0 and periodicity

    Arguments
        probabilities (paddle.tensor [shape=(1, 360, time / hop_length)])
            The probabilities for each pitch bin inferred by the network
        fmin (float)
            The minimum allowable frequency in Hz
        fmax (float)
            The maximum allowable frequency in Hz
        viterbi (bool)
            Whether to use viterbi decoding
        return_harmonicity (bool) [DEPRECATED]
            Whether to also return the network confidence
        return_periodicity (bool)
            Whether to also return the network confidence

    Returns
        pitch (paddle.tensor [shape=(1, 1 + int(time // hop_length))])
        periodicity (paddle.tensor [shape=(1, 1 + int(time // hop_length))])
    """
    # Sampling is non-differentiable, so remove from graph
    probabilities = probabilities.detach()

    # Convert frequency range to pitch bin range
    minidx = paddlecrepe.convert.frequency_to_bins(paddle.to_tensor(fmin))
    maxidx = paddlecrepe.convert.frequency_to_bins(paddle.to_tensor(fmax),
                                                  paddle.ceil)

    # Remove frequencies outside of allowable range
    probabilities[:, :minidx] = -float('inf')
    probabilities[:, maxidx:] = -float('inf')

    # Perform argmax or viterbi sampling
    bins, pitch = decoder(probabilities)

    # Deprecate return_harmonicity
    if return_harmonicity:
        message = (
            'The paddlecrepe return_harmonicity argument is deprecated and '
            'will be removed in a future release. Please use '
            'return_periodicity. Rationale: if network confidence measured '
            'harmonics, the value would be low for non-harmonic, periodic '
            'sounds (e.g., sine waves). But this is not observed.')
        warnings.warn(message, DeprecationWarning)
        return_periodicity = return_harmonicity

    if not return_periodicity:
        return pitch

    # Compute periodicity from probabilities and decoded pitch bins
    return pitch, periodicity(probabilities, bins)


def preprocess(audio,
               sample_rate,
               hop_length=None,
               batch_size=None,
               pad=True):
    """Convert audio to model input

    Arguments
        audio (paddle.tensor [shape=(1, time)])
            The audio signals
        sample_rate (int)
            The sampling rate in Hz
        hop_length (int)
            The hop_length in samples
        batch_size (int)
            The number of frames per batch
        pad (bool)
            Whether to zero-pad the audio

    Returns
        frames (paddle.tensor [shape=(1 + int(time // hop_length), 1024)])
    """
    # Default hop length of 10 ms
    hop_length = sample_rate // 100 if hop_length is None else hop_length

    # Resample
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)

    # Get total number of frames

    # Maybe pad
    if pad:
        total_frames = 1 + int(audio.shape[1] // hop_length)
        audio = paddle.nn.functional.pad(
            paddle.unsqueeze(audio, 0),
            (WINDOW_SIZE // 2, WINDOW_SIZE // 2), data_format='NCL')
        paddle.squeeze(audio, 0)
    else:
        total_frames = 1 + int((audio.shape[1] - WINDOW_SIZE) // hop_length)

    # Default to running all frames in a single batch
    batch_size = total_frames if batch_size is None else batch_size

    # Generate batches
    for i in range(0, total_frames, batch_size):

        # Batch indices
        start = max(0, i * hop_length)
        end = min(audio.shape[1],
                  (i + batch_size - 1) * hop_length + WINDOW_SIZE)

        # Chunk
        frames = paddle.nn.functional.unfold(
            audio[:, None, start:end],
            kernel_sizes=[1, WINDOW_SIZE],
            strides=[1, hop_length])

        # shape=(1 + int(time / hop_length, 1024)
        frames = frames.transpose((0, 2, 1)).reshape((-1, WINDOW_SIZE))

        # Mean-center
        frames -= frames.mean(axis=1, keepdim=True)

        # Scale
        # Note: during silent frames, this produces very large values. But
        # this seems to be what the network expects.
        frames /= paddle.maximum(paddle.to_tensor(1e-10, dtype=paddle.float32),
                            frames.std(axis=1, keepdim=True))

        yield frames


###############################################################################
# Utilities
###############################################################################


def periodicity(probabilities, bins):
    """Computes the periodicity from the network output and pitch bins"""
    # shape=(batch * time / hop_length, 360)
    probs_stacked = probabilities.transpose((0, 2, 1)).reshape((-1, PITCH_BINS))

    # shape=(batch * time / hop_length, 1)
    bins_stacked = bins.reshape((-1, 1)).cast(paddle.int64)

    # Use maximum logit over pitch bins as periodicity
    periodicity = probs_stacked.gather(bins_stacked, 1)

    # shape=(batch, time / hop_length)
    return periodicity.reshape((probabilities.shape[0], probabilities.shape[2]))


def resample(audio, sample_rate):
    """Resample audio"""

    # Convert to numpy
    audio = audio.detach().cpu().numpy().squeeze(0)

    # Resample
    # We have to use resampy if we want numbers to match Crepe
    audio = resampy.resample(audio, sample_rate, SAMPLE_RATE)

    # Convert to pypaddle
    return paddle.to_tensor(audio).unsqueeze(0)
