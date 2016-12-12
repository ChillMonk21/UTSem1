#-----------------------------------------------------------------------------
#   Default user .cshrc 
#     (last changed Sept 20, 2005)
#
#   This is an important setup file for your account. If you decide to change 
#   this file, keep a working copy until you are certain that your changes 
#   work.
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

if ( -d /home/projects/Modules/init ) then
    #unsetenv PATH MANPATH

    #
    # This works for csh *and* tcsh.
    #
    source /home/projects/Modules/init/csh

    #
    # This line determines how your modules are set up.  To change this,
    # you should use the following commands at the prompt:
    # 
    #   module initadd <module name>
    #   module initrm <moudule name>
    #
    # Try "man module" for more information on the module command.
    #
    module load ece-defaults

 
    ##

 else
    echo ''
    echo The modules directory does not currently exist on this machine.
    echo Contact the Systems Administrators Immediately!
    echo ''
endif
  
#-----------------------------------------------------------------------------
#   Shell variable customization...
#

if ($?prompt) then
  #
  #   This is an interactive shell - set aliases and shell variables here.
  #

  # 
  #   Basic prompt strings.  See tcsh(1) for more information.
  #
  if ( $?tcsh ) then 
    set prompt="%n@%m (%~) % "
    #set prompt="%n@%B%m%b (%B%~%b) % "
    set prompt2 = "%R loop: "
    set prompt3 = "oops\041 %R (y|n|e)? "
  else
    set prompt="`whoami`@`hostname` % "
  endif

  # Set up aliases works for tcsh and csh
  # edit the .csh_aliases file in your directory to add new aliases
  # by keeping aliases in this file if/when you run resetenv your
  # personal settings will be saved. This file is automatically created by
  #    "module load aliases"
  if ( -f ~/.csh_aliases ) then
     source ~/.csh_aliases
  endif

  #
  # 
  #
  set addsuffix autocorrect autoexpand autolist chase_symlinks
  set history = 100
  set noclobber filec nobeep
  set symlinks=chase
  set correct = cmd
  set fignore = (.aux .cp .dvi .elc .fn .log .o .orig .pg .toc .tp .vr .bak '~')

  #
  #   You should keep this setting!
  #
  set rmstar

  #
  # The pager is used by man, mail, elm, etc to show you files.  ("more" is 
  # the default, if you don't set PAGER.)
  #
  setenv PAGER less
  
  #
  # See man less for more choices for settings.
  #
  setenv LESS eMs

  #
  # Many people prefer emacs to vi, so you might want to change these to suit
  # your taste.
  #
  setenv EDITOR vi
  setenv VISUAL vi
else
  #
  # Not interactive.  This is either a shell script or an rsh.
  #
  unset history
  unset savehist
  unset noclobber
endif

#
# The End!
#-----------------------------------------------------------------------------
