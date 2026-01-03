# -*- coding: utf-8 -*-
"""
                                              ,
                                            ,o
                                            :o
                   _....._                  `:o
                 .'       ``-.                \o
                /  _      _   \                \o
               :  /*\    /*\   )                ;o
               |  \_/    \_/   /                ;o
               (       U      /                 ;o
                \  (\_____/) /                  /o
                 \   \_m_/  (                  /o
                  \         (                ,o:
                  )          \\,           .o;o'           ,o'o'o.
                ./          /\\o;o,,,,,;o;o;''         _,-o,-'''-o:o.
 .             ./o./)        \    'o'o'o''         _,-'o,o'         o
 o           ./o./ /       .o \\.              __,-o o,o'
 \o.       ,/o /  /o/)     | o o'-..____,,-o'o o_o-'
 `o:o...-o,o-' ,o,/ |     \   'o.o_o_o_o,o--''
 .,  ``o-o'  ,.oo/   'o /\\.o`.
 `o`o-....o'o,-'   /o /   \o \\.                       ,o..         o
   ``o-o.o--      /o /      \o.o--..          ,,,o-o'o.--o:o:o,,..:o
                 (oo(          `--o.o`o---o'o'o,o,-'''        o'o'o
                  \ o\              ``-o-o''''
   ,-o;o           \o \
  /o/               )o )
 (o(               /o /                |
  \o\.       ...-o'o /             \   |
    \o`o`-o'o o,o,--'       ~~~~~~~~\~~|~~~~~The Kraken 2025~~~~~~~~~~~~~~~~~~~~~~~
      ```o--'''                       \| /
                                       |/
 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                       |
 ~~~~~~~ August 2025 with Grok4 & ChatGPT5 (mostly Grok)~~~~~~~~~~~~~~~~~~~~~~~~~~

 <SCRIPT>: <ONE-LINE PURPOSE/TAGLINE>
"""

# Tiny helper route: list installed LoRAs for the optional searchable UI
from aiohttp import web
import folder_paths
from server import PromptServer

@PromptServer.instance.routes.get("/kraken/loras")
async def kraken_list_loras(request):
    try:
        items = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
    except Exception:
        items = []
    return web.json_response({"items": items})
