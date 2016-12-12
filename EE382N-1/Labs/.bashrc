# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi
#---------------Display settting for user and root environment-----------------
if [ "$BASH" ]; then
		PS1='\[\033[1;32m\]\u\[\033[0;31m\]@\[\033[1;32m\]\h\[\033[0;37m\]:\[\033[1;36m\]\w\[\033[0;31m\] > '
else
	if [ "`id -u`" -eq 0 ]; then
		PS1='# '
	else
		PS1='$ '
	fi
fi
export PS1
PS1="$PS1"'$([ -n "$TMUX" ] && tmux setenv TMUXPWD_$(tmux display -p "#D" | tr -d %) "$PWD")'
export CLICOLOR='true'
export LC_ALL="en_US" 
#export LSCOLORS=ExFxCxDxBxegedabagacad 
export LSCOLORS=GxBxCxDxBxegedabagacad
#export LSCOLORS=exfxcxdxbxegedabagacad
#export LSCOLORS=gxfxcxdxbxegedabagacad 
#export LSCOLORS=HxFxCxDxBxegedabagacad

export PATH="$PATH:/opt/local/bin:/opt/local/sbin"
alias d0="du -h -d 0"
alias d1="du -h -d 1"
alias d2="du -h -d 2"
alias ffdisk="diskutil list"
#-------To increase history size------------#
export HISTSIZE=10000
export HISTCONTROL=erasedups
shopt -s histappend
export TERM=xterm-color
export GREP_OPTIONS="--color=auto" 
export GREP_COLOR="1;32"
unset GREP_OPTIONS
unset GREP_COLOR
alias sgrep="find . | grep -i $@"
alias c="clear"
alias vi="vim"
alias cim="vim"
alias bim="vim"
alias l="ls -lA"
alias ssh="ssh -X"
alias ll="ls -alhSv"
alias lsr="ls -lSv"
alias md="mkdir"
alias rd="rmdir"
alias cd..="cd .."
alias ..="cd .."
alias ...="cd ../.."
alias v="vim ~/.ssh/known_hosts"
alias fd="find . -name"
alias net='lsof -P -i -n | cut -f 1 -d " " |uniq'
alias wa="~/Dropbox/Git/wallpaper.sh"
#------------------------------------------------------------------------------
alias HEX="ruby -e 'printf(\"0x%X\n\", ARGV[0])'"
alias DEC="ruby -e 'printf(\"%d\n\", ARGV[0])'"
alias BIN="ruby -e 'printf(\"%bb\n\", ARGV[0])'"
alias WORD="ruby -e 'printf(\"0x%04X\n\", ARGV[0])'"
#------------------- Security-----------------//
alias rm="rm -i"
alias rmf="rm -irf"
alias grep="/usr/bin/grep $GREP_OPTIONS"
alias cp="cp -i"
alias mv="mv -i"
alias ln="ln -i"
ccat(){
   	pygmentize -f terminal -g $@
}
man() {
    env \
    LESS_TERMCAP_mb=$(printf "\e[1;31m") \
    LESS_TERMCAP_md=$(printf "\e[1;31m") \
    LESS_TERMCAP_me=$(printf "\e[0m") \
    LESS_TERMCAP_se=$(printf "\e[0m") \
    LESS_TERMCAP_so=$(printf "\e[1;44;33m") \
    LESS_TERMCAP_ue=$(printf "\e[0m") \
    LESS_TERMCAP_us=$(printf "\e[1;32m") \
    man "$@"
}
