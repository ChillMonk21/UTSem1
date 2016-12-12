# This is the default .profile file for the ECE Learning Resource Center.
# It is intended to work with ksh, but should work with any bourne compatible 
# shell (such as sh or bash).  
#

#-----------------------------------------------------------------------------
#   The umask sets the access permissions for any new files you create.
#   Common umasks:
#     077 - removes all permissions except the owner's
#     022 - the standard unix default - removes write permissions for
#           everyone other than the owner.  (Allows others to read most
#           of your files.)
#     027 - removes write permissions for the members of a file's group,
#           and removes all permissions for all others.
#   For an explanation of the octal encoding, see "man chmod".
#

umask 077


#-----------------------------------------------------------------------------
#   Modules are an easy, flexible way to customize your environment.
#   To see what modules are available on the system, type "module avail".
#   Using modules replaces setting your "path" environment variable.

if [ -d /home/projects/Modules/init ]; then

  #
  # This works for sh, ksh and bash 
  # except for the set-alias cmd which doesn't work in sh. 
  # only sort of works in ksh --
  # doesn't work at login. one has to specifically execute a bash shell
  #
  . /home/projects/Modules/init/bash

  #
  # This line determines how your modules are set up.  To change this,
  # you should use the following commands at the prompt:
  #
  #   module initadd <module name>
  #   module initrm <moudule name>
  #
  #
  module load ece-defaults

else
    echo ''
    echo The modules directory does not currently exist on this machine.
    echo Contact the Systems Administrators Immediately!
    echo ''
fi

#
# Keep aliases in your .bash_aliases file they will work for ksh and sh
# and bash. By keeping aliases in these files if/when you run resetenv they 
# will not be overwritten. This file is automatically created by 
#  	"module load aliases"  
# 
if [ -f ~/.bash_aliases ]; then
	. ~/.bash_aliases
fi

#
# Set the Prompt based on user and hostname
#
USER=`whoami`
HOSTNAME=`hostname`
BASE=`basename $SHELL`

if [ "$BASE" == "ksh" ]; then
   PS1='$USER@${HOSTNAME%%.*} (${PWD##$HOME/}) % '
   PS2="loop: "
   export PS1 PS2
   # Fix a few KSH issues with the arrow commands 
   # Allows ksh to act like Bash
   set -o emacs
   alias \!\!='fc -e -'
   # Fix the backspace key
   stty erase \^H erase \^? kill \^U intr \^C
else
   # Works for BASH and SH
   PS1="$USER@${HOSTNAME%%.*} (\w) % "
   PS2="loop: "
   export PS1 PS2
fi

# 
# set the path to default mailbox
#
MAIL=/var/spool/mail/$USER
MAILPATH=$MAIL
export MAIL MAILPATH


#
# This will set the default editor to emacs.  If you want this feature, 
# uncomment the next lines
#
#if [ -x /usr/local/bin/emacs ]; then
#   export EDITOR=emacs
#fi


# End .profile
