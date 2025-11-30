"""
Guacamol/Molecule benchmarks using SELFIES-VAE latent space.

Modified to be BoTorch compatible from https://github.com/nataliemaus/aabo
"""

import logging
import urllib
from math import log
from pathlib import Path

import selfies as sf
import torch
from botorch.test_functions import SyntheticTestFunction
from guacamol import standard_benchmarks
from rdkit import Chem
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import Dataset


med1 = standard_benchmarks.median_camphor_menthol()  # 'Median molecules 1'
med2 = standard_benchmarks.median_tadalafil_sildenafil()  # 'Median molecules 2',
pdop = standard_benchmarks.perindopril_rings()  # 'Perindopril MPO',
osmb = standard_benchmarks.hard_osimertinib()  # 'Osimertinib MPO',
adip = standard_benchmarks.amlodipine_rings()  # 'Amlodipine MPO'
siga = standard_benchmarks.sitagliptin_replacement()  # 'Sitagliptin MPO'
zale = standard_benchmarks.zaleplon_with_other_formula()  # 'Zaleplon MPO'
valt = standard_benchmarks.valsartan_smarts()  # 'Valsartan SMARTS',
dhop = standard_benchmarks.decoration_hop()  # 'Deco Hop'
shop = standard_benchmarks.scaffold_hop()  # Scaffold Hop'
rano = standard_benchmarks.ranolazine_mpo()  # 'Ranolazine MPO'
fexo = standard_benchmarks.hard_fexofenadine()  # 'Fexofenadine MPO'... 'make fexofenadine less greasy'

guacamol_objs = {
    "med1": med1,
    "pdop": pdop,
    "adip": adip,
    "rano": rano,
    "osmb": osmb,
    "siga": siga,
    "zale": zale,
    "valt": valt,
    "med2": med2,
    "dhop": dhop,
    "shop": shop,
    "fexo": fexo,
}

DEFAULT_SELFIES_VOCAB = [
    "<start>",
    "<stop>",
    "[#Branch1]",
    "[#Branch2]",
    "[#C-1]",
    "[#C]",
    "[#N+1]",
    "[#N]",
    "[#O+1]",
    "[=B]",
    "[=Branch1]",
    "[=Branch2]",
    "[=C-1]",
    "[=C]",
    "[=N+1]",
    "[=N-1]",
    "[=NH1+1]",
    "[=NH2+1]",
    "[=N]",
    "[=O+1]",
    "[=OH1+1]",
    "[=O]",
    "[=PH1]",
    "[=P]",
    "[=Ring1]",
    "[=Ring2]",
    "[=S+1]",
    "[=SH1]",
    "[=S]",
    "[=Se+1]",
    "[=Se]",
    "[=Si]",
    "[B-1]",
    "[BH0]",
    "[BH1-1]",
    "[BH2-1]",
    "[BH3-1]",
    "[B]",
    "[Br+2]",
    "[Br-1]",
    "[Br]",
    "[Branch1]",
    "[Branch2]",
    "[C+1]",
    "[C-1]",
    "[CH1+1]",
    "[CH1-1]",
    "[CH1]",
    "[CH2+1]",
    "[CH2]",
    "[C]",
    "[Cl+1]",
    "[Cl+2]",
    "[Cl+3]",
    "[Cl-1]",
    "[Cl]",
    "[F+1]",
    "[F-1]",
    "[F]",
    "[H]",
    "[I+1]",
    "[I+2]",
    "[I+3]",
    "[I]",
    "[N+1]",
    "[N-1]",
    "[NH0]",
    "[NH1+1]",
    "[NH1-1]",
    "[NH1]",
    "[NH2+1]",
    "[NH3+1]",
    "[N]",
    "[O+1]",
    "[O-1]",
    "[OH0]",
    "[O]",
    "[P+1]",
    "[PH1]",
    "[PH2+1]",
    "[P]",
    "[Ring1]",
    "[Ring2]",
    "[S+1]",
    "[S-1]",
    "[SH1]",
    "[S]",
    "[Se+1]",
    "[Se-1]",
    "[SeH1]",
    "[SeH2]",
    "[Se]",
    "[Si-1]",
    "[SiH1-1]",
    "[SiH1]",
    "[SiH2]",
    "[Si]",
]


def is_valid_molecule(x):
    """
    Artifact from AABO.
    """
    return True


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    dim: int = -1,
    return_randoms: bool = False,
    randoms: Tensor = None,
) -> Tensor:
    """
    Mostly from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    """
    if randoms is None:
        randoms = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
    gumbels = (logits + randoms) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if return_randoms:
        return ret, randoms
    else:
        return ret


class PositionalEncoding(nn.Module):
    """
    Custom positional encoding module.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Adds positional encoding to the input tensor and applies dropout."""
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class SELFIESDataset(Dataset):
    """
    SELFIES dataset.
    """

    def __init__(
        self,
        fname=None,
        load_data=False,
    ):
        self.data = []
        if load_data:
            assert fname is not None
            with open(fname, "r") as f:
                selfie_strings = [x.strip() for x in f.readlines()]
            for string in selfie_strings:
                self.data.append(list(sf.split_selfies(string)))
            self.vocab = {token for selfie in self.data for token in selfie}
            self.vocab.discard(".")
            self.vocab = ["<start>", "<stop>", *sorted(list(self.vocab))]  # noqa: C414
        else:
            self.vocab = DEFAULT_SELFIES_VOCAB

        self.vocab2idx = {v: i for i, v in enumerate(self.vocab)}

    def tokenize_selfies(self, selfies_list):
        """
        Tokenizes a list of SELFIES (Self-Referencing Embedded Strings).

        Args:
            selfies_list (list[str]): A list of SELFIES strings to be tokenized.

        Returns:
            list[list[str]]: A list where each element is a list of tokens derived
            from the corresponding SELFIES string in the input.
        """
        tokenized_selfies = []
        for string in selfies_list:
            tokenized_selfies.append(list(sf.split_selfies(string)))
        return tokenized_selfies

    def encode(self, smiles):
        """Encodes a list of SMILES strings into a list of token indices."""
        return torch.tensor([self.vocab2idx[s] for s in [*smiles, "<stop>"]])

    def decode(self, tokens):
        """
        Decodes a list of token indices into their corresponding vocabulary string representation.

        Parameters:
            tokens : list[int]
                A list of token indices to decode.

        Returns:
            str
                The decoded string representation of the input tokens.

        Raises:
            ValueError: If the decoding logic encounters an unsupported or unexpected scenario such
                        as missing a valid token sequence structure.
        """
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index("<stop>") if "<stop>" in dec else None  # want first stop token
        selfie = dec[0:stop]  # cut off stop tokens
        while "<start>" in selfie:  # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = 1 + dec.index("<start>")
            selfie = selfie[start:]
        selfie = "".join(selfie)
        return selfie

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the tokenized SMILES string at the given index."""
        return self.encode(self.data[idx])

    @property
    def vocab_size(self):
        """Returns the size of the vocabulary."""
        return len(self.vocab)


class InfoTransformerVAE(nn.Module):
    """
    SELFIES VAE
    """

    def __init__(
        self,
        dataset: SELFIESDataset,
        bottleneck_size: int = 2,
        d_model: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
    ):
        super().__init__()

        assert bottleneck_size is not None, (
            "Dont set bottleneck_size to None. Unbounded sequences dont support this yet"
        )

        self.max_string_length = 256

        self.dataset = dataset
        self.vocab_size = len(self.dataset.vocab)

        self.bottleneck_size = bottleneck_size
        self.d_model = d_model
        self.is_autoencoder = is_autoencoder

        self.kl_factor = kl_factor

        self.min_posterior_std = min_posterior_std
        encoder_embedding_dim = 2 * d_model

        self.encoder_token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=encoder_embedding_dim)
        self.encoder_position_encoding = PositionalEncoding(
            encoder_embedding_dim, dropout=encoder_dropout, max_len=5_000
        )
        self.decoder_token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=d_model)
        self.decoder_position_encoding = PositionalEncoding(d_model, dropout=decoder_dropout, max_len=5_000)
        self.decoder_token_unembedding = nn.Parameter(torch.randn(d_model, self.vocab_size))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_embedding_dim,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_feedforward,
                dropout=encoder_dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=encoder_num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=decoder_nhead,
                dim_feedforward=decoder_dim_feedforward,
                dropout=decoder_dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=decoder_num_layers,
        )

    def sample_prior(self, n):
        """Samples from the prior distribution"""
        if self.bottleneck_size is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = self.bottleneck_size

        return torch.randn(n, sequence_length, self.d_model).to(self.device)

    def sample_posterior(self, mu, sigma, n=None):
        """Samples from the posterior distribution given the mean and standard deviation"""
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)

        return mu + torch.randn_like(mu) * sigma

    def generate_pad_mask(self, tokens):
        """Generate mask that tells encoder to ignore all but first stop token"""
        mask = tokens == 1
        inds = mask.float().argmax(dim=-1)  # Returns first index along axis when multiple present
        mask[torch.arange(0, tokens.shape[0]), inds] = False
        return mask

    def encode(self, tokens, as_probs=False):
        """
        Encodes the input sequence into latent space representations.

        Parameters:
        - tokens: Tensor
          Input sequence of tokens.
        - as_probs: bool, optional
          Whether to treat input tokens as probabilities.

        Returns:
        - Tuple[Tensor, Tensor]
          Returns two tensors: the mean (mu) and standard deviation (sigma) of the latent
          representation.
        """
        if as_probs:
            embed = tokens @ self.encoder_token_embedding.weight
        else:
            embed = self.encoder_token_embedding(tokens)

        embed = self.encoder_position_encoding(embed)

        pad_mask = self.generate_pad_mask(tokens)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        mu = encoding[..., : self.d_model]
        sigma = functional.softplus(encoding[..., self.d_model :]) + self.min_posterior_std

        if self.bottleneck_size is not None:
            mu = mu[:, : self.bottleneck_size, :]
            sigma = sigma[:, : self.bottleneck_size, :]

        return mu, sigma

    def decode(self, z, tokens, as_probs=False):
        """
        Decodes the input latent variable and token sequences to produce output logits.

        Parameters:
            z: torch.Tensor
                The latent variable representing encoded input features.
            tokens: torch.Tensor
                The token sequences to be decoded. The tokens provided should include one element
                less than the full sequence, as embedding processing skips the last token.
            as_probs: bool, optional
                If True, the token sequences are expected to be probabilities and a weighted
                embedding computation is performed. Defaults to False.

        Returns:
            torch.Tensor
                The logits resulting from the decoding process, which can be directly used
                for further tasks such as classification or sequence generation.
        """
        if as_probs:
            embed = tokens[:, :-1] @ self.decoder_token_embedding.weight
        else:
            embed = self.decoder_token_embedding(tokens[:, :-1])

        embed = torch.cat(
            [
                # Zero is the start token
                torch.zeros(embed.shape[0], 1, embed.shape[-1], device=self.device),
                embed,
            ],
            dim=1,
        )
        embed = self.decoder_position_encoding(embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(embed.shape[1]).to(self.device)
        decoding = self.decoder(tgt=embed, memory=z, tgt_mask=tgt_mask)
        logits = decoding @ self.decoder_token_unembedding

        return logits

    @torch.no_grad()
    def sample(self, n: int = -1, z: Tensor = None, differentiable: bool = False, return_logits: bool = False):
        """
        Samples sequences based on input latent variables or prior sampling.

        Parameters:
        n: int
            The number of samples to generate.
        z: Tensor, optional
            A tensor representing the latent variables used for decoding. If None, the function will sample from the
            prior distribution.
        differentiable: bool
            If True, the function returns differentiable results. Defaults to False.
        return_logits: bool
            If True, returns the logits alongside the sampled results. Defaults to False.

        Returns:
        Tensor or Tuple[Tensor, Tensor]

        Raises:
        - RuntimeError: If any internal operations fail due to dimension mismatches or computational issues.
        """
        model_state = self.training
        self.eval()
        if z is None:
            z = self.sample_prior(n)
        else:
            n = z.shape[0]

        tokens = torch.zeros(n, 1, device=self.device).long()  # Start token is 0, stop token is 1
        random_gumbels = torch.zeros(n, 0, self.vocab_size, device=self.device)
        while True:  # Loop until every molecule hits a stop token
            tgt = self.decoder_token_embedding(tokens)
            tgt = self.decoder_position_encoding(tgt)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[-1]).to(self.device)

            decoding = self.decoder(tgt=tgt, memory=z, tgt_mask=tgt_mask)
            logits = decoding @ self.decoder_token_unembedding
            sample, randoms = gumbel_softmax(logits, dim=-1, hard=True, return_randoms=True)

            tokens = torch.cat([tokens, sample[:, -1, :].argmax(dim=-1)[:, None]], dim=-1)
            random_gumbels = torch.cat([random_gumbels, randoms], dim=1)

            # 1 is the stop token. Check if all molecules have a stop token in them
            if (
                torch.all((tokens == 1).sum(dim=-1) > 0).item() or tokens.shape[-1] > self.max_string_length
            ):  # no longer break at 1024, instead variable max string lengtth
                break

        self.train(model_state)
        if not differentiable:
            sample = tokens

        if return_logits:
            return sample, logits
        else:
            return sample

    def is_valid(self, x):
        """
        Determines if the input data represents valid molecules.

        Arguments:
        x: Tensor
            A tensor containing encoded representations of molecules on a specific
            device.

        Returns:
        Tensor
            A tensor of dtype `torch.float`, containing validity flags for each
            molecule, and located on the same device as the input.
        """
        device = x.device
        x = x.cpu()
        v = [is_valid_molecule(self.dataset.decode(s)) for s in x]

        return torch.tensor(v, dtype=torch.float, device=device)

    def forward(self, tokens):
        """
        Computes the forward pass for a variational autoencoder model.

        Parameters:
            tokens: Tensor
                Input tokens to be processed by the model.

        Returns:
            dict:
                A dictionary containing the following keys:
                - "loss": The total loss combining reconstruction loss and KL divergence.
                - "z": The latent representation from the posterior or encoder.
                - "recon_loss": The reconstruction loss calculated from logits and tokens.
                - "kldiv": The KL divergence calculated from the prior and posterior.
                - "recon_token_acc": Token-level reconstruction accuracy.
                - "recon_string_acc": String-level reconstruction accuracy.
                - "sigma_mean": The mean value of the sigma parameter.
        """
        mu, sigma = self.encode(tokens)

        if self.is_autoencoder:
            z = mu
        else:
            z = self.sample_posterior(mu, sigma)

        logits = self.decode(z, tokens)

        recon_loss = functional.cross_entropy(
            logits.permute(0, 2, 1), tokens, reduction="none"
        ).mean()  # .sum(1).mean(0)

        # No need for KL divergence when \alpha = 1
        # see https://ojs.aaai.org//index.php/AAAI/article/view/4538 Eq. 6
        # Equation from the original "Auto-Encoding Variational Bayes" paper: https://arxiv.org/pdf/1312.6114.pdf
        sigma2 = sigma.pow(2)
        kldiv = 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1).mean()  # .sum(dim=(1, 2)).mean(0)

        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv
        loss = primary_loss

        return {
            "loss": loss,
            "z": z,
            "recon_loss": recon_loss,
            "kldiv": kldiv,
            "recon_token_acc": (logits.argmax(dim=-1) == tokens).float().mean(),
            "recon_string_acc": (logits.argmax(dim=-1) == tokens).all(dim=1).float().mean(dim=0),
            "sigma_mean": sigma.mean(),
        }


class GuacamolObjective(SyntheticTestFunction):
    """Guacamol optimization tasks
    https://github.com/BenevolentAI/guacamol,
    Using LS-BO with SELFIES VAE from LOL-BO
    """

    def __init__(
        self,
        guacamol_task_id,
        noise_std: float = 0,
        negate: bool = False,  # Defaults to maximization
    ):
        # search space dim
        self.dim = 256  # Default VAE dimension
        self.dtype = torch.float32
        # absolute upper and lower bounds on search space
        # lb = (-8,)  # based on forwarding 20k guacamol molecules through vae and seeing min of zs -6.3683
        # ub = (8,)  # based on forwarding 20k guacamol molecules through vae and seeing max of zs 7.2140
        self._bounds = [(-8, 8) for _ in range(self.dim)]
        self.continuous_inds = list(range(self.dim))
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

        self.path_to_vae_statedict = Path(__file__).parent.parent.parent / Path(
            "./data/selfies_vae/selfies-vae-state-dict.pt"
        )
        if not self.path_to_vae_statedict.exists():
            self.path_to_vae_statedict.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/nataliemaus/aabo/raw/refs/heads/main/tasks/utils/selfies_vae/selfies-vae-state-dict.pt"
            logging.info(f"Downloading {url} to {self.path_to_vae_statedict}")
            urllib.request.urlretrieve(url, self.path_to_vae_statedict)

        self.max_string_length = 128
        self.guacamol_obj_func = guacamol_objs[guacamol_task_id].objective
        self.initialize_vae()

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        """Function defines batched function f(x) (the function we want to optimize).

        Args:
            X (enumerable): (bsz, dim) enumerable tye of length equal to batch size (bsz),
            each item in enumerable type must be a float tensor of shape (dim,)
            (each is a vector in input search space).

        Returns:
            tensor: (bsz, 1) float tensor giving reward obtained by passing each x in xs into f(x).
        """
        smiles_list = self.vae_decode(z=X)
        ys = []
        for smile in smiles_list:
            y = self.smile_to_guacamole_score(smile=smile)
            if y is None:
                ys.append(-0.01)
            else:
                ys.append(y)

        return torch.tensor(ys).to(X)

    def smile_to_guacamole_score(self, smile):
        """Converts SMILES string to guacamole score"""
        if smile is None or len(smile) == 0:
            return None
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        score = self.guacamol_obj_func.score(smile)
        if score is None:
            return None
        if score < 0:
            return None
        return score

    def initialize_vae(self):
        """Sets self.vae to the desired pretrained vae and
        sets self.dataobj to the corresponding data class
        used to tokenize inputs, etc."""
        self.dataobj = SELFIESDataset()
        self.vae = InfoTransformerVAE(dataset=self.dataobj)

        # load in state dict of trained model:
        if self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict, weights_only=True)
            self.vae.load_state_dict(state_dict, strict=True)
        self.vae = self.vae.eval()
        # set max string length that VAE can generate
        self.vae.max_string_length = self.max_string_length

    def vae_decode(self, z):
        """Input
            z: a tensor latent space points
        Output
            a corresponding list of the decoded input space
            items output by vae decoder
        """
        self.vae.eval()
        # sample molecular string form VAE decoder
        with torch.no_grad():
            sample = self.vae.sample(z=z.reshape(-1, 2, 128))
        # grab decoded selfies strings
        decoded_selfies = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # decode selfies strings to smiles strings (SMILES is needed format for oracle)
        decoded_smiles = []
        for selfie in decoded_selfies:
            smile = sf.decoder(selfie)
            decoded_smiles.append(smile)

        return decoded_smiles

    def to(self, dtype, device):
        """Override to() method to also move VAE to the specified device"""
        result = super().to(dtype=dtype, device=device)
        if hasattr(self, "vae") and self.vae is not None:
            self.vae.device = device
        return result


if __name__ == "__main__":
    obj = GuacamolObjective(guacamol_task_id="rano").to("cuda")
    x = torch.randn(12, 256).to(dtype=obj.dtype).to("cuda")
    y = obj(x)
    # print(f"y: {y}")
