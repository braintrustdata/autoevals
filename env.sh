SRC_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

ps1_old="$PS1"
source $SRC_ROOT/venv/bin/activate
export PS1="(autoevals) $ps1_old"
