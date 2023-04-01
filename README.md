![Anya from SPY x FAMILY pointing at a television screen with Rogue displayed saying 'I like this show.'](LOVEIT.png)

# Chizuru
Chizuru is an AI that plays the 1980 computer game Rogue.
While this repository contains the code for the AI, it also contains the dissertation released alongside this code in `writeup/`.

You can learn more about Rogue on the [NetHack Wiki page](https://nethackwiki.com/wiki/Rogue_(game)) about it.

## Setup
This thing is designed to run in a Docker container. To do that, run these:
```shell
docker build -t chizuru .
docker run
```
After that, it should be "smooth" sailing.

## Files
Chizuru saves its training checkpoints to `czr-xxxx.ckpt` where `xxxx` is the epoch number.

## Bugs
Probably infinite (although countably infinite). However, the distant screams of your PC running this model is *not* a bug. It's a feature.

## Licence
This program is released under the GNU General Public Licence v3.0.

You should have received a copy of the GNU General Public Licence
along with this program. If not, see <https://www.gnu.org/licenses/>.
