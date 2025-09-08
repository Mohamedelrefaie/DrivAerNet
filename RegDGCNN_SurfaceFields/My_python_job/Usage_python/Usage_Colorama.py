from colorama import Fore, Style

    logging.info(f"{Fore.GREEN}all_metrics: {all_metrics}{Style.RESET_ALL}")
    -> {Style.RESET_ALL}
       -> Reset the default color for the subsequent info

from colorama import Fore

Fore.BLACK
Fore.RED
Fore.GREEN
Fore.YELLOW
Fore.BLUE
Fore.MAGENTA
Fore.CYAN
Fore.WHITE
Fore.RESET    # Reset to default

from colorama import Back

Back.BLACK
Back.RED
Back.GREEN
Back.YELLOW
Back.BLUE
Back.MAGENTA
Back.CYAN
Back.WHITE
Back.RESET    # Reset to default

